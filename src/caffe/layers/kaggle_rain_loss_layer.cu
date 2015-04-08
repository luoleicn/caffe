#include <vector>

#include "caffe/layer.hpp"
#include "caffe/util/io.hpp"
#include "caffe/util/math_functions.hpp"
#include "caffe/vision_layers.hpp"

namespace caffe {

template <typename Dtype>
void KaggleRainLossLayer<Dtype>::Forward_gpu(const vector<Blob<Dtype>*>& bottom,
    const vector<Blob<Dtype>*>& top) {

  int num = bottom[0]->num();
  Dtype* h_func_data = h_func_.mutable_gpu_data();
  caffe_gpu_set(num * 70, Dtype(0), h_func_data);

  CUDA_KERNEL_LOOP(i, num) {
      int rain  = bottom[1]->data_at(i, 0, 0, 0);
      caffe_gpu_set(70 - rain, Dtype(1), h_func_data + i*70 + rain);
  }

  int count = bottom[0]->count();
  caffe_gpu_sub(
      count, 
      bottom[0]->gpu_data(), 
      h_func_.gpu_data(), 
      diff_.mutable_gpu_data());

  Dtype dot;
  caffe_gpu_dot(count, diff_.gpu_data(), diff_.gpu_data(), &dot);
  Dtype loss = dot / bottom[0]->num() / Dtype(70);
  top[0]->mutable_gpu_data()[0] = loss;
}

template <typename Dtype>
void KaggleRainLossLayer<Dtype>::Backward_gpu(const vector<Blob<Dtype>*>& top,
    const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom) {

  if (propagate_down[0]) {
      const Dtype alpha = top[0]->cpu_diff()[0] / bottom[0]->num();
      caffe_gpu_axpby(
          bottom[0]->count(),              // count
          alpha,                              // alpha
          diff_.gpu_data(),                   // a
          Dtype(0),                           // beta
          bottom[0]->mutable_gpu_diff());  // b
  }
}

INSTANTIATE_LAYER_GPU_FUNCS(KaggleRainLossLayer);

}  // namespace caffe
