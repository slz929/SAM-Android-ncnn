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

#ifndef YOLOX_H
#define YOLOX_H

#include <opencv2/core/core.hpp>

#include <net.h>

//////// sam
#include "segment_anything.h"

struct Object
{
    cv::Rect_<float> rect;
    int label;
    float prob;
   
};


class Yolox
{
public:
    Yolox();

    int load(const char* modeltype, int target_size, const float* mean_vals, const float* norm_vals, bool use_gpu = false);

    int load(AAssetManager* mgr, const char* modeltype, int target_size, const float* mean_vals, const float* norm_vals, bool use_gpu = false);

    int detect(const cv::Mat& rgb, std::vector<Object>& objects, float prob_threshold = 0.45f, float nms_threshold = 0.65f);

    int draw(cv::Mat& rgb, const std::vector<Object>& objects);
    ////////////////////////////////////////////////
    //sam
    int Init(AAssetManager* mgr, const char* modeltype_decoder, const char* modeltype_encoder, int target_size, const float* mean_vals, const float* norm_vals, bool use_gpu = false);
    int ImageEmbedding(const cv::Mat& bgr, pipeline_result_t& pipeline_result);
    int Predict(const cv::Mat& bgr, pipeline_result_t& pipeline_result);
    int AutoPredict(const cv::Mat& bgr, pipeline_result_t& pipeline_result, int n_per_side = 32);
    void Draw(const cv::Mat& bgr, const pipeline_result_t& pipeline_result);
private:
    void get_grid_points(std::vector<float>& points_xy_vec, int n_per_side);
    std::shared_ptr<SegmentAnything> sam_;
// private:

    ncnn::Net yolox;

    int target_size;
    float mean_vals[3];
    float norm_vals[3];
    int image_w;
    int image_h;
    int in_w;
    int in_h;

    ncnn::UnlockedPoolAllocator blob_pool_allocator;
    ncnn::PoolAllocator workspace_pool_allocator;
};

#endif // NANODET_H
