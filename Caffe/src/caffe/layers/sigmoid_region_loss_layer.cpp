#include <algorithm>
#include <cfloat>
#include <vector>

#include "caffe/util/math_functions.hpp"
#include "caffe/layers/sigmoid_region_loss_layer.hpp"

namespace caffe {

template <typename Dtype>
void SigmoidRegionLossLayer<Dtype>::LayerSetUp(
    const vector<Blob<Dtype>*>& bottom, const vector<Blob<Dtype>*>& top) {
  LossLayer<Dtype>::LayerSetUp(bottom, top);
  sigmoid_bottom_vec_.clear();
  sigmoid_bottom_vec_.push_back(bottom[0]);
  sigmoid_top_vec_.clear();
  sigmoid_top_vec_.push_back(sigmoid_output_.get());
  sigmoid_layer_->SetUp(sigmoid_bottom_vec_, sigmoid_top_vec_);

  const SigmoidRegionLossParameter  sigmoid_region_loss_param = this->layer_param_.sigmoid_region_loss_param();
  reg_coeff_ = sigmoid_region_loss_param.lambda();
	kernel_size_ = sigmoid_region_loss_param.kernel_size();
}

template <typename Dtype>
void SigmoidRegionLossLayer<Dtype>::Reshape(
    const vector<Blob<Dtype>*>& bottom, const vector<Blob<Dtype>*>& top) {
  LossLayer<Dtype>::Reshape(bottom, top);
  CHECK_EQ(bottom[0]->count(), bottom[1]->count()) <<
      "SIGMOID_REGION_LOSS layer inputs must have the same count.";
  sigmoid_layer_->Reshape(sigmoid_bottom_vec_, sigmoid_top_vec_);
  reg_info_.Reshape(1,1,1,bottom[0]->count());
}

template <typename Dtype>
void SigmoidRegionLossLayer<Dtype>::Forward_cpu(
    const vector<Blob<Dtype>*>& bottom, const vector<Blob<Dtype>*>& top) {
  sigmoid_bottom_vec_[0] = bottom[0];
  sigmoid_layer_->Forward(sigmoid_bottom_vec_, sigmoid_top_vec_);
  const Dtype* target = bottom[1]->cpu_data();
  const Dtype* pr = sigmoid_output_->cpu_data();
  Dtype *reg_loss = reg_info_.mutable_cpu_data();

  Dtype loss_pos = 0;
  Dtype loss_neg = 0;
  Dtype count_pos = 0;
  Dtype count_neg = 0;

  /* step 1: compute pixel-wise loss */
  int dim = bottom[0]->count() / bottom[0]->num();
  for (int j = 0; j < dim; j++) {
    if (target[j] == 1) {
      count_pos++;
	    loss_pos -= log(std::max(pr[j], Dtype(FLT_MIN)));
	  } else if (target[j] == 0) {
	    count_neg++;
	    loss_neg -= log(std::max(Dtype(1.0 - pr[j]), Dtype(FLT_MIN)));
	  } else {/* */}
  }

  /*
   * step 2: compute region-wise loss 
   * for a given pixel p_i, its corresponding region-wise loss is
   * defined as:
   *	Region(p_i) = \lambda (Mean(N_i(p_i)) - y_i)^2
   * where N_i(p_i) denotes the neighbours of p_i, and Mean denotes
   * average probability of these neighbours.
   */
  const int width = bottom[0]->width();
  const int height = bottom[0]->height();
  Dtype mean_neibour = 0;
	int margin = (kernel_size_ - 1) / 2;
	for (int w = margin; w < width - margin; w++) {
		for (int h = margin; h < height - margin; h++) { /* loop over pixel*/
			mean_neibour = 0;
			for (int r = w - margin; r <= w + margin; r++) { /* a kernel */
				for (int c = h - margin; c <= h + margin; c++) {
					if ((r == w) && (c == h)) { continue; }
				  mean_neibour += pr[width*r+c]; /* sum over inside kernel*/
				}
			}
			mean_neibour /= (kernel_size_ * kernel_size_ - 1);
	    if (target[width*(w)+(h)] == 1) {
		    loss_pos += reg_coeff_ * (mean_neibour - 1) * (mean_neibour - 1);
		    reg_loss[width*(w)+(h)] = (mean_neibour -1);
	    } else if (target[width*(w)+(h)] == 0) {
		    loss_neg += reg_coeff_ * (mean_neibour * mean_neibour);
		    reg_loss[width*(w)+(h)] = (mean_neibour);
	    } else { /* */ }
		}
	}
#if 0
  const int width = bottom[0]->width();
  const int height = bottom[0]->height();
  Dtype mean_neibour = 0;
  for (int w = 1; w < width -1; w++) {
	for (int h = 1; h < height - 1; h++) {
      mean_neibour = pr[width*(w-1)+(h-1)]+pr[width*(w-1)+(h)]+
					 pr[width*(w-1)+(h+1)]+pr[width*(w)+(h-1)]+
					 pr[width*(w)+(h+1)]+pr[width*(w+1)+(h-1)]+
					 pr[width*(w+1)+(h)]+pr[width*(w+1)+(h+1)];
	  mean_neibour /= 8;
	  if (target[width*(w)+(h)] == 1)  {
		loss_pos += reg_coeff_ * (mean_neibour - 1) * (mean_neibour - 1);
		reg_loss[width*(w)+(h)] = (mean_neibour -1);
	  } else if (target[width*(w)+(h)] == 0) {
		loss_neg += reg_coeff_ * (mean_neibour * mean_neibour);
		reg_loss[width*(w)+(h)] = (mean_neibour);
	  } else { /* */ }
	}
  }
#endif

  /* step 3: weight balanced */
  weight_pos_ = 1.0 * count_neg / (count_pos + count_neg);
  weight_neg_ = 1.0 * count_pos / (count_pos + count_neg);
  loss_pos *= weight_pos_;
  loss_neg *= weight_neg_;

  top[0]->mutable_cpu_data()[0] = (loss_pos + loss_neg);
}

template <typename Dtype>
void SigmoidRegionLossLayer<Dtype>::Backward_cpu(
    const vector<Blob<Dtype>*>& top, const vector<bool>& propagate_down,
    const vector<Blob<Dtype>*>& bottom) {
  if (propagate_down[1]) {
    LOG(FATAL) << this->type()
               << " Layer cannot backpropagate to label inputs.";
  }
  if (propagate_down[0]) {
    const int count = bottom[0]->count();
    const int width = bottom[0]->width();
    const int height = bottom[0]->height();

    const Dtype* pr = sigmoid_output_->cpu_data();
    const Dtype* target = bottom[1]->cpu_data();
	const Dtype* reg_loss = reg_info_.cpu_data();
    Dtype* bottom_diff = bottom[0]->mutable_cpu_diff();

	/* step 1: unary gradient */
    caffe_sub(count, pr, target, bottom_diff);

	/* 
	 * step 2: region gradient
	 * 2a) compute neighbours' gradients 
	 * 2b) accumulate unary gradient and neighbours gradients
	 */
#if 0
    for (int w = 1; w < width -1; w++) {
      for (int h = 1; h < height - 1; h++) {
		Dtype grad_temp1 = reg_loss[width*(w-1)+(h-1)];
		Dtype grad_temp2 = reg_loss[width*(w-1)+(h)];
		Dtype grad_temp3 = reg_loss[width*(w-1)+(h+1)];
		Dtype grad_temp4 = reg_loss[width*(w)+(h-1)];
		Dtype grad_temp5 = reg_loss[width*(w)+(h+1)];
		Dtype grad_temp6 = reg_loss[width*(w+1)+(h-1)];
		Dtype grad_temp7 = reg_loss[width*(w+1)+(h)];
		Dtype grad_temp8 = reg_loss[width*(w+1)+(h+1)];

		Dtype reg_diff = (grad_temp1+grad_temp2+grad_temp3+grad_temp4+grad_temp5+grad_temp6+grad_temp7+grad_temp8)*(pr[width*(w)+(h)])*(1.0-pr[width*(w)+(h)])*reg_coeff_;
		bottom_diff[width*(w)+(h)] += reg_diff;
      }
	}
#endif
	int margin = (kernel_size_ - 1) / 2;
	for (int w = margin; w < width - margin; w++) {
		for (int h = margin; h < height - margin; h++) { /* loop over pixel*/
			Dtype reg_diff = 0;
			for (int r = w - margin; r <= w + margin; r++) { /* a kernel */
				for (int c = h - margin; c <= h + margin; c++) {
					if ((r == w) && (c == h)) { continue; }
					reg_diff += reg_loss[width*r+c];
				}
			}
			reg_diff *= pr[width*w+h]*(1.0-pr[width*w+h])*reg_coeff_;
			bottom_diff[width*w+h] += reg_diff;
		}
	}
	
	/* step 3: weight balanced */
    const int dim = bottom[0]->count() / bottom[0]->num();
	for (int j = 0; j < dim; j++) {
	  if (target[j] == 1) {
		bottom_diff[j] *= weight_pos_;
	  } else if (target[j] == 0) {
		bottom_diff[j] *= weight_neg_;
	  } else {/* */}
	}

    const Dtype loss_weight = top [0]->cpu_diff()[0];
    caffe_scal(count, loss_weight, bottom_diff);
  }
}

#ifdef CPU_ONLY
STUB_GPU_BACKWARD(SigmoidRegionLossLayer, Backward);
#endif

INSTANTIATE_CLASS(SigmoidRegionLossLayer);
REGISTER_LAYER_CLASS(SigmoidRegionLoss);

}  // namespace caffe
