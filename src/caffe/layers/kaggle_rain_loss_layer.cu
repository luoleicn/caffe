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
  Dtype* h_func_data = h_func_.mutable_gpu_data();

  for (int i = 0; i < num; i ++) {
      int rain  = bottom[1]->data_at(i, 0, 0, 0);
      //LOG(INFO) << "rain i=" << i << " rain=" << rain;
      caffe_gpu_set(rain, Dtype(0), h_func_data + i*70);
      caffe_gpu_set(70 - rain, Dtype(1), h_func_data + i*70 + rain);
  }

  Dtype* cdf = cdf_.mutable_cpu_data();
  //memset(cdf, Dtype(1), cdf->count()*sizeof(Dtype));
  for (int i = 0; i < num; i ++) {
      Dtype last(0);
      for (int j = 0; j < 70; j ++) {
	      if (j <= 10) {
		      cdf[i*70+j] = bottom[0]->data_at(i, j, 0, 0) + last;
		      last = cdf[i*70+j];
	      }
	      else {
		      cdf[i*70+j] = Dtype(1);
	      }
	      //LOG(INFO) << "i=" << i << " j=" << j << " cdf=" << cdf[i*70+j];
      }
  }


  int count = cdf_.count();
  caffe_gpu_sub(
      count, 
      cdf_.gpu_data(), 
      h_func_.gpu_data(), 
      diff_.mutable_gpu_data());

//  for (int i = 0; i < num; i ++) {
//      for (int j = 0; j < 70; j ++) {
//	      if (j <= 10) {
//		      LOG(INFO) << "i=" << i << " j=" << j << " cdf=" << cdf[i*70+j]
//			      << " h_func_=" << h_func_.data_at(i, j, 0, 0)
//			      << " diff=" << diff_.data_at(i, j, 0, 0);
//	      }
//      }
//  }

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
      //caffe_gpu_axpby(
      //    bottom[0]->count(),              // count
      //    alpha,                              // alpha
      //    diff_.gpu_data(),                   // a
      //    Dtype(0),                           // beta
      //    bottom[0]->mutable_gpu_diff());  // b

      Dtype* ret_diff = bottom[0]->mutable_cpu_diff();
      int num = bottom[0]->num();
      for (int i = 0; i < num; i ++) {
	      Dtype last(0);
	      for (int j = 69; j >= 0; j --) {
		      if (j > 10) {
			      continue;
		      }
		      else {
			      ret_diff[i*10+j] = last + diff_.data_at(i, j, 0, 0);
			      ret_diff[i*10+j] *= alpha;

			      last += diff_.data_at(i, j, 0, 0);
		      }
	      }
      }

  }
}

INSTANTIATE_LAYER_GPU_FUNCS(KaggleRainLossLayer);

}  // namespace caffe
