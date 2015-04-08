#include <vector>

#include "caffe/layer.hpp"
#include "caffe/util/io.hpp"
#include "caffe/util/math_functions.hpp"
#include "caffe/vision_layers.hpp"

namespace caffe {

//template <typename Dtype>
//void KaggleRainLossLayer<Dtype>::LayerSetUp(
//    const vector<Blob<Dtype>*>& bottom, const vector<Blob<Dtype>*>& top) {
//
//  LossLayer<Dtype>::LayerSetUp(bottom, top);
//
//  capacity_ = 100
//  h_func_vec_.resize(capacity_);
//  vector<int> mult_dims(1, 70);
//  for (int i = 0; i < capacity_; i ++) {
//      h_func_vec_[i] = shared_ptr<Blob<Dtype> >(new Blob<Dtype>());
//      h_func_vec_[i]->Reshape(mult_dims);
//  }
//}

template <typename Dtype>
void KaggleRainLossLayer<Dtype>::Reshape(
  const vector<Blob<Dtype>*>& bottom, const vector<Blob<Dtype>*>& top) {

  LossLayer<Dtype>::Reshape(bottom, top);

  vector<int> shape(4);
  shape[0] = bottom[0]->num();
  shape[1] = 70;
  shape[2] = 1;
  shape[3] = 1;
  h_func_.Reshape(shape);
  cdf_.Reshape(shape);

  diff_.ReshapeLike(*bottom[0]);
}

template <typename Dtype>
void KaggleRainLossLayer<Dtype>::Forward_cpu(const vector<Blob<Dtype>*>& bottom,
    const vector<Blob<Dtype>*>& top) {

  Dtype* h_func_data = h_func_.mutable_cpu_data();
  int num = bottom[0]->num();
  caffe_set(num * 70, Dtype(0), h_func_data);
  for (int i = 0; i < num; i ++) {
      int rain  = bottom[1]->data_at(i, 0, 0, 0);
      caffe_set(70 - rain, Dtype(1), h_func_data + i*70 + rain);
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
  caffe_sub(
      count, 
      cdf_.cpu_data(),
      h_func_.cpu_data(), 
      diff_.mutable_cpu_data());

  Dtype dot = caffe_cpu_dot(count, diff_.cpu_data(), diff_.cpu_data());
  Dtype loss = dot / bottom[0]->num() / Dtype(70);
  top[0]->mutable_cpu_data()[0] = loss;
}

template <typename Dtype>
void KaggleRainLossLayer<Dtype>::Backward_cpu(const vector<Blob<Dtype>*>& top,
    const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom) {

  if (propagate_down[0]) {
      const Dtype alpha = top[0]->cpu_diff()[0] / bottom[0]->num();
      caffe_cpu_axpby(
          bottom[0]->count(),              // count
          alpha,                              // alpha
          diff_.cpu_data(),                   // a
          Dtype(0),                           // beta
          bottom[0]->mutable_cpu_diff());  // b
  }
}

#ifdef CPU_ONLY
STUB_GPU(KaggleRainLossLayer);
#endif

INSTANTIATE_CLASS(KaggleRainLossLayer);
REGISTER_LAYER_CLASS(KaggleRainLoss);

}  // namespace caffe
