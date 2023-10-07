// Tencent is pleased to support the open source community by making ncnn available.
//
// Copyright (C) 2021 THL A29 Limited, a Tencent company. All rights reserved.
//
// Licensed under the BSD 3-Clause License (the "License"); you may not use this file except
// in compliance with the License. You may obtain a copy of the License at
//
// https://opensource.org/licenses/BSD-3-Clause
//
// Unless required by applicable law or agreed to in writing, software distributed
// under the License is distributed on an "AS IS" BASIS, WITHOUT WARRANTIES OR
// CONDITIONS OF ANY KIND, either express or implied. See the License for the
// specific language governing permissions and limitations under the License.

#include "sam.h"

#include <opencv2/core/core.hpp>
#include <opencv2/imgproc/imgproc.hpp>

#include "cpu.h"



// YOLOX use the same focus in yolov5
class YoloV5Focus : public ncnn::Layer
{
public:
    YoloV5Focus()
    {
        one_blob_only = true;
    }

    virtual int forward(const ncnn::Mat& bottom_blob, ncnn::Mat& top_blob, const ncnn::Option& opt) const
    {
        int w = bottom_blob.w;
        int h = bottom_blob.h;
        int channels = bottom_blob.c;

        int outw = w / 2;
        int outh = h / 2;
        int outc = channels * 4;

        top_blob.create(outw, outh, outc, 4u, 1, opt.blob_allocator);
        if (top_blob.empty())
            return -100;

        #pragma omp parallel for num_threads(opt.num_threads)
        for (int p = 0; p < outc; p++)
        {
            const float* ptr = bottom_blob.channel(p % channels).row((p / channels) % 2) + ((p / channels) / 2);
            float* outptr = top_blob.channel(p);

            for (int i = 0; i < outh; i++)
            {
                for (int j = 0; j < outw; j++)
                {
                    *outptr = *ptr;

                    outptr += 1;
                    ptr += 2;
                }

                ptr += w;
            }
        }

        return 0;
    }
};

DEFINE_LAYER_CREATOR(YoloV5Focus)


struct GridAndStride
{
    int grid0;
    int grid1;
    int stride;
};

static inline float intersection_area(const Object& a, const Object& b)
{
    cv::Rect_<float> inter = a.rect & b.rect;
    return inter.area();
}

static void qsort_descent_inplace(std::vector<Object>& faceobjects, int left, int right)
{
    int i = left;
    int j = right;
    float p = faceobjects[(left + right) / 2].prob;

    while (i <= j)
    {
        while (faceobjects[i].prob > p)
            i++;

        while (faceobjects[j].prob < p)
            j--;

        if (i <= j)
        {
            // swap
            std::swap(faceobjects[i], faceobjects[j]);

            i++;
            j--;
        }
    }

    #pragma omp parallel sections
    {
        #pragma omp section
        {
            if (left < j) qsort_descent_inplace(faceobjects, left, j);
        }
        #pragma omp section
        {
            if (i < right) qsort_descent_inplace(faceobjects, i, right);
        }
    }
}

static void qsort_descent_inplace(std::vector<Object>& objects)
{
    if (objects.empty())
        return;

    qsort_descent_inplace(objects, 0, objects.size() - 1);
}

static void nms_sorted_bboxes(const std::vector<Object>& faceobjects, std::vector<int>& picked, float nms_threshold)
{
    picked.clear();

    const int n = faceobjects.size();

    std::vector<float> areas(n);
    for (int i = 0; i < n; i++)
    {
        areas[i] = faceobjects[i].rect.area();
    }

    for (int i = 0; i < n; i++)
    {
        const Object& a = faceobjects[i];

        int keep = 1;
        for (int j = 0; j < (int)picked.size(); j++)
        {
            const Object& b = faceobjects[picked[j]];

            // intersection over union
            float inter_area = intersection_area(a, b);
            float union_area = areas[i] + areas[picked[j]] - inter_area;
            // float IoU = inter_area / union_area
            if (inter_area / union_area > nms_threshold)
                keep = 0;
        }

        if (keep)
            picked.push_back(i);
    }
}
static void generate_grids_and_stride(const int target_w, const int target_h, std::vector<int>& strides, std::vector<GridAndStride>& grid_strides)
{
	for (auto stride : strides)
	{
		int num_grid_w = target_w / stride;
		int num_grid_h = target_h / stride;
		for (int g1 = 0; g1 < num_grid_h; g1++)
		{
			for (int g0 = 0; g0 < num_grid_w; g0++)
			{
				GridAndStride gs;
				gs.grid0 = g0;
				gs.grid1 = g1;
				gs.stride = stride;
				grid_strides.push_back(gs);
			}
		}
	}
}

static void generate_yolox_proposals(std::vector<GridAndStride> grid_strides, const ncnn::Mat& feat_blob, float prob_threshold, std::vector<Object>& objects)
{
    const int num_grid = feat_blob.h;
    fprintf(stderr, "output height: %d, width: %d, channels: %d, dims:%d\n", feat_blob.h, feat_blob.w, feat_blob.c, feat_blob.dims);

    const int num_class = feat_blob.w - 5;

    const int num_anchors = grid_strides.size();

    const float* feat_ptr = feat_blob.channel(0);
    for (int anchor_idx = 0; anchor_idx < num_anchors; anchor_idx++)
    {
        const int grid0 = grid_strides[anchor_idx].grid0;
        const int grid1 = grid_strides[anchor_idx].grid1;
        const int stride = grid_strides[anchor_idx].stride;

        // yolox/models/yolo_head.py decode logic
        //  outputs[..., :2] = (outputs[..., :2] + grids) * strides
        //  outputs[..., 2:4] = torch.exp(outputs[..., 2:4]) * strides
        float x_center = (feat_ptr[0] + grid0) * stride;
        float y_center = (feat_ptr[1] + grid1) * stride;
        float w = exp(feat_ptr[2]) * stride;
        float h = exp(feat_ptr[3]) * stride;
        float x0 = x_center - w * 0.5f;
        float y0 = y_center - h * 0.5f;

        float box_objectness = feat_ptr[4];
        for (int class_idx = 0; class_idx < num_class; class_idx++)
        {
            float box_cls_score = feat_ptr[5 + class_idx];
            float box_prob = box_objectness * box_cls_score;
            if (box_prob > prob_threshold)
            {
                Object obj;
                obj.rect.x = x0;
                obj.rect.y = y0;
                obj.rect.width = w;
                obj.rect.height = h;
                obj.label = class_idx;
                obj.prob = box_prob;

                objects.push_back(obj);
            }

        } // class loop
        feat_ptr += feat_blob.w;

    } // point anchor loop
}
 
 
Yolox::Yolox()
{
    blob_pool_allocator.set_size_compare_ratio(0.f);
    workspace_pool_allocator.set_size_compare_ratio(0.f);
}

int Yolox::load(const char* modeltype, int _target_size, const float* _mean_vals, const float* _norm_vals, bool use_gpu)
{
    yolox.clear();
    blob_pool_allocator.clear();
    workspace_pool_allocator.clear();

    ncnn::set_cpu_powersave(2);
    ncnn::set_omp_num_threads(ncnn::get_big_cpu_count());

    yolox.opt = ncnn::Option();

#if NCNN_VULKAN
    yolox.opt.use_vulkan_compute = use_gpu;
#endif
    yolox.register_custom_layer("YoloV5Focus", YoloV5Focus_layer_creator);
    yolox.opt.num_threads = ncnn::get_big_cpu_count();
    yolox.opt.blob_allocator = &blob_pool_allocator;
    yolox.opt.workspace_allocator = &workspace_pool_allocator;

    char parampath[256];
    char modelpath[256];
    sprintf(parampath, "%s.param", modeltype);
    sprintf(modelpath, "%s.bin", modeltype);

    yolox.load_param(parampath);
    yolox.load_model(modelpath);

    target_size = _target_size;
    mean_vals[0] = _mean_vals[0];
    mean_vals[1] = _mean_vals[1];
    mean_vals[2] = _mean_vals[2];
    norm_vals[0] = _norm_vals[0];
    norm_vals[1] = _norm_vals[1];
    norm_vals[2] = _norm_vals[2];

    return 0;
}

int Yolox::load(AAssetManager* mgr, const char* modeltype, int _target_size, const float* _mean_vals, const float* _norm_vals, bool use_gpu)
{
    yolox.clear();
    blob_pool_allocator.clear();
    workspace_pool_allocator.clear();

    ncnn::set_cpu_powersave(2);
    ncnn::set_omp_num_threads(ncnn::get_big_cpu_count());

    yolox.opt = ncnn::Option();
#if NCNN_VULKAN
    yolox.opt.use_vulkan_compute = use_gpu;
#endif
    yolox.register_custom_layer("YoloV5Focus", YoloV5Focus_layer_creator);
    yolox.opt.num_threads = ncnn::get_big_cpu_count();
    yolox.opt.blob_allocator = &blob_pool_allocator;
    yolox.opt.workspace_allocator = &workspace_pool_allocator;

    char parampath[256];
    char modelpath[256];
    sprintf(parampath, "%s.param", modeltype);
    sprintf(modelpath, "%s.bin", modeltype);

    yolox.load_param(mgr, parampath);
    yolox.load_model(mgr, modelpath);


    target_size = _target_size;
    mean_vals[0] = _mean_vals[0];
    mean_vals[1] = _mean_vals[1];
    mean_vals[2] = _mean_vals[2];
    norm_vals[0] = _norm_vals[0];
    norm_vals[1] = _norm_vals[1];
    norm_vals[2] = _norm_vals[2];

    return 0;
}


int Yolox::detect(const cv::Mat& rgb, std::vector<Object>& objects, float prob_threshold, float nms_threshold)
{

    int img_w = rgb.cols;
    int img_h = rgb.rows;

    // letterbox pad to multiple of 32
    int w = img_w;
    int h = img_h;
    float scale = 1.f;
    if (w > h)
    {
        scale = (float)target_size / w;
        w = target_size;
        h = h * scale;
    }
    else
    {
        scale = (float)target_size / h;
        h = target_size;
        w = w * scale;
    }

    ncnn::Mat in = ncnn::Mat::from_pixels_resize(rgb.data, ncnn::Mat::PIXEL_RGB, img_w, img_h, w, h);

    // pad to target_size rectangle
    // yolov5/utils/datasets.py letterbox
    int wpad = (w + 31) / 32 * 32 - w;
    int hpad = (h + 31) / 32 * 32 - h;
    ncnn::Mat in_pad;
    ncnn::copy_make_border(in, in_pad, 0, hpad, 0, wpad, ncnn::BORDER_CONSTANT, 114.f);

    // so for 0-255 input image, rgb_mean should multiply 255 and norm should div by std.
    // new release of yolox has deleted this preprocess,if you are using new release please don't use this preprocess.
    in_pad.substract_mean_normalize(mean_vals, norm_vals);

    ncnn::Extractor ex = yolox.create_extractor();

    ex.input("images", in_pad);

    std::vector<Object> proposals;

    {
        ncnn::Mat out;
        ex.extract("output", out);

        std::vector<int> strides = {8, 16, 32}; // might have stride=64
        std::vector<GridAndStride> grid_strides;
        generate_grids_and_stride(in_pad.w, in_pad.h, strides, grid_strides);
        generate_yolox_proposals(grid_strides, out, prob_threshold, proposals);
    }

    // sort all proposals by score from highest to lowest
    qsort_descent_inplace(proposals);

    // apply nms with nms_threshold
    std::vector<int> picked;
    nms_sorted_bboxes(proposals, picked, nms_threshold);

    int count = picked.size();

    objects.resize(count);
    for (int i = 0; i < count; i++)
    {
        objects[i] = proposals[picked[i]];

        // adjust offset to original unpadded
        float x0 = (objects[i].rect.x) / scale;
        float y0 = (objects[i].rect.y) / scale;
        float x1 = (objects[i].rect.x + objects[i].rect.width) / scale;
        float y1 = (objects[i].rect.y + objects[i].rect.height) / scale;

        // clip
        x0 = std::max(std::min(x0, (float)(img_w - 1)), 0.f);
        y0 = std::max(std::min(y0, (float)(img_h - 1)), 0.f);
        x1 = std::max(std::min(x1, (float)(img_w - 1)), 0.f);
        y1 = std::max(std::min(y1, (float)(img_h - 1)), 0.f);

        objects[i].rect.x = x0;
        objects[i].rect.y = y0;
        objects[i].rect.width = x1 - x0;
        objects[i].rect.height = y1 - y0;
    }

    return 0;
}

int Yolox::draw(cv::Mat& rgb, const std::vector<Object>& objects)
{
    static const char* class_names[] = {
            "person", "bicycle", "car", "motorcycle", "airplane", "bus", "train", "truck", "boat", "traffic light",
            "fire hydrant", "stop sign", "parking meter", "bench", "bird", "cat", "dog", "horse", "sheep", "cow",
            "elephant", "bear", "zebra", "giraffe", "backpack", "umbrella", "handbag", "tie", "suitcase", "frisbee",
            "skis", "snowboard", "sports ball", "kite", "baseball bat", "baseball glove", "skateboard", "surfboard",
            "tennis racket", "bottle", "wine glass", "cup", "fork", "knife", "spoon", "bowl", "banana", "apple",
            "sandwich", "orange", "broccoli", "carrot", "hot dog", "pizza", "donut", "cake", "chair", "couch",
            "potted plant", "bed", "dining table", "toilet", "tv", "laptop", "mouse", "remote", "keyboard", "cell phone",
            "microwave", "oven", "toaster", "sink", "refrigerator", "book", "clock", "vase", "scissors", "teddy bear",
            "hair drier", "toothbrush"
    };
    static const unsigned char colors[19][3] = {
        { 54,  67, 244},
        { 99,  30, 233},
        {176,  39, 156},
        {183,  58, 103},
        {181,  81,  63},
        {243, 150,  33},
        {244, 169,   3},
        {212, 188,   0},
        {136, 150,   0},
        { 80, 175,  76},
        { 74, 195, 139},
        { 57, 220, 205},
        { 59, 235, 255},
        {  7, 193, 255},
        {  0, 152, 255},
        { 34,  87, 255},
        { 72,  85, 121},
        {158, 158, 158},
        {139, 125,  96}
    };

    int color_index = 0;

    for (size_t i = 0; i < objects.size(); i++)
    {
        const Object& obj = objects[i];

        const unsigned char* color = colors[color_index % 19];
        color_index++;

        cv::Scalar cc(color[0], color[1], color[2]);

        cv::rectangle(rgb,obj.rect, cc, 2);

        char text[256];
        sprintf(text, "%s %.1f%%", class_names[obj.label], obj.prob * 100);

        int baseLine = 0;
        cv::Size label_size = cv::getTextSize(text, cv::FONT_HERSHEY_SIMPLEX, 0.5, 1, &baseLine);

        int x = obj.rect.x;
        int y = obj.rect.y - label_size.height - baseLine;
        if (y < 0)
            y = 0;
        if (x + label_size.width > rgb.cols)
            x = rgb.cols - label_size.width;

        cv::rectangle(rgb, cv::Rect(cv::Point(x, y), cv::Size(label_size.width, label_size.height + baseLine)), cc, -1);

        cv::Scalar textcc = (color[0] + color[1] + color[2] >= 381) ? cv::Scalar(0, 0, 0) : cv::Scalar(255, 255, 255);

        cv::putText(rgb, text, cv::Point(x, y + label_size.height), cv::FONT_HERSHEY_SIMPLEX, 0.5, textcc, 1);

    }
    
    
    return 0;
}


int Yolox::Init(AAssetManager* mgr, const char* modeltype_decoder, const char* modeltype_encoder, int _target_size, const float* _mean_vals, const float* _norm_vals, bool use_gpu)
{
    yolox.clear();
    blob_pool_allocator.clear();
    workspace_pool_allocator.clear();

    ncnn::set_cpu_powersave(2);
    ncnn::set_omp_num_threads(ncnn::get_big_cpu_count());

    yolox.opt = ncnn::Option();
#if NCNN_VULKAN
    yolox.opt.use_vulkan_compute = use_gpu;
#endif
//    yolox.register_custom_layer("YoloV5Focus", YoloV5Focus_layer_creator);
//    yolox.opt.num_threads = ncnn::get_big_cpu_count();
//    yolox.opt.blob_allocator = &blob_pool_allocator;
//    yolox.opt.workspace_allocator = &workspace_pool_allocator;

    char parampath_decode[256];
    char modelpath_decode[256];
    char parampath_encode[256];
    char modelpath_encode[256];
    sprintf(parampath_decode, "%s.param", modeltype_decoder);
    sprintf(modelpath_decode, "%s.bin", modeltype_decoder);
    sprintf(parampath_encode, "%s.param", modeltype_encoder);
    sprintf(modelpath_encode, "%s.bin", modeltype_encoder);
    std::string a= parampath_decode;
    std::string b= modelpath_decode;
    std::string c= parampath_encode;
    std::string d= modelpath_encode;


    //////sam
    sam_ = std::make_shared<SegmentAnything>();
    int ret = sam_->Load(c,d,a,b);
    if (ret<0){
        return -1;
    }

//    target_size = _target_size;
//    mean_vals[0] = _mean_vals[0];
//    mean_vals[1] = _mean_vals[1];
//    mean_vals[2] = _mean_vals[2];
//    norm_vals[0] = _norm_vals[0];
//    norm_vals[1] = _norm_vals[1];
//    norm_vals[2] = _norm_vals[2];

    return ret;
}

int Yolox::ImageEmbedding(const cv::Mat& bgr, pipeline_result_t& pipeline_result)
{
//    std::cout << "start image encoder..." << std::endl;
    sam_->ImageEncoder(bgr, pipeline_result.image_embeddings, pipeline_result.image_info);
//    std::cout << "finish image encoder..." << std::endl;

    return 0;
}

int Yolox::AutoPredict(const cv::Mat& bgr, pipeline_result_t& pipeline_result, int n_per_side)
{
    pipeline_result.prompt_info.prompt_type = PromptType::Point;

    //generate grid points
    std::vector<float> points_xy_vec;
    get_grid_points(points_xy_vec, n_per_side);

    std::vector<sam_result_t> proposals;
    for(int i = 0; i < n_per_side; ++i) {
        std::vector<sam_result_t> objects;
        for(int j = 0; j < n_per_side; ++j) {
            pipeline_result.prompt_info.points.clear();
            pipeline_result.prompt_info.labels.clear();
            pipeline_result.prompt_info.points.push_back(points_xy_vec[i * n_per_side * 2 + 2 * j] * pipeline_result.image_info.img_w);
            pipeline_result.prompt_info.points.push_back(points_xy_vec[i * n_per_side * 2 + 2 * j + 1] * pipeline_result.image_info.img_h);
            
            pipeline_result.prompt_info.points.push_back(0);
            pipeline_result.prompt_info.points.push_back(0);

            pipeline_result.prompt_info.labels.push_back(1);
            pipeline_result.prompt_info.labels.push_back(-1);

            sam_->MaskDecoder(pipeline_result.image_embeddings, pipeline_result.image_info, pipeline_result.prompt_info, objects);
        }
        proposals.insert(proposals.end(), objects.begin(), objects.end());
//        std::cout<<"processing: "<< i <<"/"<<n_per_side<<std::endl;
    }

    std::vector<int> picked;
    sam_->NMS(bgr, proposals, picked);
    int num_picked = picked.size();
    
    for(int j = 0; j < num_picked; ++j){
        pipeline_result.sam_result.push_back(proposals[picked[j]]);
    }
    
    return 0;
}


int Yolox::Predict(const cv::Mat& bgr, pipeline_result_t& pipeline_result)
{
    sam_->MaskDecoder(pipeline_result.image_embeddings, pipeline_result.image_info, pipeline_result.prompt_info, pipeline_result.sam_result);
    return 0;
}


void Yolox::Draw(const cv::Mat& bgr, const pipeline_result_t& pipeline_result)
{
    static const unsigned char colors[81][3] = {
            {56,  0,   255},
            {226, 255, 0},
            {0,   94,  255},
            {0,   37,  255},
            {0,   255, 94},
            {255, 226, 0},
            {0,   18,  255},
            {255, 151, 0},
            {170, 0,   255},
            {0,   255, 56},
            {255, 0,   75},
            {0,   75,  255},
            {0,   255, 169},
            {255, 0,   207},
            {75,  255, 0},
            {207, 0,   255},
            {37,  0,   255},
            {0,   207, 255},
            {94,  0,   255},
            {0,   255, 113},
            {255, 18,  0},
            {255, 0,   56},
            {18,  0,   255},
            {0,   255, 226},
            {170, 255, 0},
            {255, 0,   245},
            {151, 255, 0},
            {132, 255, 0},
            {75,  0,   255},
            {151, 0,   255},
            {0,   151, 255},
            {132, 0,   255},
            {0,   255, 245},
            {255, 132, 0},
            {226, 0,   255},
            {255, 37,  0},
            {207, 255, 0},
            {0,   255, 207},
            {94,  255, 0},
            {0,   226, 255},
            {56,  255, 0},
            {255, 94,  0},
            {255, 113, 0},
            {0,   132, 255},
            {255, 0,   132},
            {255, 170, 0},
            {255, 0,   188},
            {113, 255, 0},
            {245, 0,   255},
            {113, 0,   255},
            {255, 188, 0},
            {0,   113, 255},
            {255, 0,   0},
            {0,   56,  255},
            {255, 0,   113},
            {0,   255, 188},
            {255, 0,   94},
            {255, 0,   18},
            {18,  255, 0},
            {0,   255, 132},
            {0,   188, 255},
            {0,   245, 255},
            {0,   169, 255},
            {37,  255, 0},
            {255, 0,   151},
            {188, 0,   255},
            {0,   255, 37},
            {0,   255, 0},
            {255, 0,   170},
            {255, 0,   37},
            {255, 75,  0},
            {0,   0,   255},
            {255, 207, 0},
            {255, 0,   226},
            {255, 245, 0},
            {188, 255, 0},
            {0,   255, 18},
            {0,   255, 75},
            {0,   255, 151},
            {255, 56,  0},
            {245, 255, 0}
    };

    cv::Mat img = bgr.clone();

    for(size_t n = 0; n < pipeline_result.sam_result.size(); ++n){
        for (int y = 0; y < img.rows; ++y) {
            uchar* image_ptr = img.ptr(y);
            const uchar* mask_ptr = pipeline_result.sam_result[n].mask.ptr<uchar>(y);
            for (int x = 0; x < img.cols; ++x) {
                if (mask_ptr[x] > 0)
                {
                    image_ptr[0] = cv::saturate_cast<uchar>(image_ptr[0] * 0.5 + colors[n][0] * 0.5);
                    image_ptr[1] = cv::saturate_cast<uchar>(image_ptr[1] * 0.5 + colors[n][1] * 0.5);
                    image_ptr[2] = cv::saturate_cast<uchar>(image_ptr[2] * 0.5 + colors[n][2] * 0.5);
                }
                image_ptr += 3;
            }
        }

        //cv::rectangle(img, pipeline_result.sam_result[n].box, cv::Scalar(0,255,0), 2, 8,0);

        switch(pipeline_result.prompt_info.prompt_type){
            case PromptType::Point:
                for(int i = 0; i < pipeline_result.prompt_info.points.size() / 2; ++i){
                    cv::circle(img, cv::Point(pipeline_result.prompt_info.points[2 * i], pipeline_result.prompt_info.points[2 * i + 1]), 5, cv::Scalar(255,255,0),2,8);
                }
                break;
            case PromptType::Box:
                cv::rectangle(img, cv::Rect(cv::Point(pipeline_result.prompt_info.points[0], pipeline_result.prompt_info.points[1]), cv::Point(pipeline_result.prompt_info.points[2], pipeline_result.prompt_info.points[3])), cv::Scalar(255,255,0),2,8);
                break;
            default:
                break;
        }
    }

    return ;
    // cv::imshow("dst", img);
    //cv::imshow("mask", pipeline_result.sam_result.mask);
    // cv::imwrite("dst.jpg",img);
    // cv::waitKey();
}

void Yolox::get_grid_points(std::vector<float>& points_xy_vec, int n_per_side)
{
    float offset = 1.f / (2 * n_per_side);
    
    float start = offset;
    float end = 1 - offset;
    float step = (end - start) / (n_per_side - 1);

    std::vector<float> points_one_side;
    for (int i = 0; i < n_per_side; ++i) {
        points_one_side.push_back(start + i * step);
    }

    points_xy_vec.resize(n_per_side * n_per_side * 2);
    for (int i = 0; i < n_per_side; ++i) {
        for (int j = 0; j < n_per_side; ++j) {
            points_xy_vec[i * n_per_side * 2 + 2 * j + 0] = points_one_side[j];
            points_xy_vec[i * n_per_side * 2 + 2 * j + 1] = points_one_side[i];
        }
    }
}
