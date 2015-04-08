#include <vector>

#include "caffe/layer.hpp"
#include "caffe/util/io.hpp"
#include "caffe/util/math_functions.hpp"
#include "caffe/vision_layers.hpp"

namespace caffe {

template <typename Dtype>
__global__ void KaggleRainLossForwardGPU(const int nthreads,
          const Dtype* bottom_data, const Dtype* h_func_data) {

  CUDA_KERNEL_LOOP(i, nthreads) {
      int rain  = (int)(bottom_data[i]);
      const Dtype d0(0); 
      const Dtype d1(1); 
      caffe_gpu_set(rain, d0, h_func_data + i*70);
      caffe_gpu_set(70 - rain, d1, h_func_data + i*70 + rain);
  }

}

template <typename Dtype>
void KaggleRainLossLayer<Dtype>::Forward_gpu(const vector<Blob<Dtype>*>& bottom,
    const vector<Blob<Dtype>*>& top) {

  int num = bottom[0]->num();
  //int nthreads = num;
  Dtype* h_func_data = h_func_.mutable_gpu_data();

//  KaggleRainLossForwardGPU<Dtype><<<CAFFE_GET_BLOCKS(nthreads),
//      CAFFE_CUDA_NUM_THREADS>>>(nthreads, bottom[1].gpu_data(), h_func_data);

  for (int i = 0; i < num; i ++) {
      int rain  = bottom[1]->data_at(i, 0, 0, 0);
      caffe_gpu_set(rain, Dtype(0), h_func_data + i*70);
      caffe_gpu_set(70 - rain, Dtype(1), h_func_data + i*70 + rain);
  }

  Dtype* cdf = cdf_.mutable_cpu_data();
  memcpy(cdf, bottom[0]->cpu_data(), bottom[0]->count()*sizeof(Dtype));

  for (int i = 0; i < num; i ++) {
      Dtype last(0);
      for (int j = 0; j < 70; j ++) {
          cdf[i*70+j] += last;
          last = cdf[i*70+j];
      }
  }

  int count = bottom[0]->count();
  caffe_gpu_sub(
      count, 
      cdf_.gpu_data(), 
      h_func_.gpu_data(), 
      diff_.mutable_gpu_data());

  Dtype dot;
  caffe_gpu_dot(count, diff_.gpu_data(), diff_.gpu_data(), &dot);
  Dtype loss = dot / bottom[0]->num() / Dtype(70);
  top[0]->mutable_cpu_data()[0] = loss;
}

template <typename Dtype>
void KaggleRainLossLayer<Dtype>::Backward_gpu(const vector<Blob<Dtype>*>& top,
    const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom) {

  if (propagate_down[0]) {
      const Dtype alpha = top[0]->cpu_diff()[0] / bottom[0]->num() / Dtype(35);
      caffe_gpu_axpby(
          bottom[0]->count(),              // count
          alpha,                              // alpha
          diff_.gpu_data(),                   // a
          Dtype(0),                           // beta
          bottom[0]->mutable_gpu_diff());  // b

      Dtype* ret_diff = bottom[0]->mutable_cpu_diff();
      Dtype last(0);
      for (int i = 69; i >= 0; i --) {
          ret_diff[i] += last;
          last = ret_diff[i];
      }

  }
}

INSTANTIATE_LAYER_GPU_FUNCS(KaggleRainLossLayer);

}  // namespace caffe
