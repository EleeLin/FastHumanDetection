#include "adaptive_manifold_filter.h"
#include <cmath>
#include <limits>
#include <opencv2/core/internal.hpp>
#include <opencv2/imgproc/imgproc.hpp>

using namespace std;
using namespace cv;

namespace
{
    // struct Buf
	struct Buf
    {
        Mat_<Point3f> eta_1;
		Mat_<float> eta_1_single_channel;
        Mat_<uchar> cluster_1;
		
        Mat_<Point3f> tilde_dst;
		Mat_<float> tilde_dst_single_channel;
        Mat_<float> alpha;
        Mat_<Point3f> diff;
		Mat_<float> diff_single_channel;
        Mat_<Point3f> dst;
		Mat_<float> dst_single_channel;

        Mat_<float> V;

        Mat_<Point3f> dIcdx;
		Mat_<float> dIcdx_single_channel;
        Mat_<Point3f> dIcdy;
		Mat_<float> dIcdy_single_channel;
        Mat_<float> dIdx;
        Mat_<float> dIdy;
        Mat_<float> dHdx;
        Mat_<float> dVdy;

        Mat_<float> t;

        Mat_<float> theta_masked;
        Mat_<Point3f> mul;
		Mat_<float> mul_single_channel;
        Mat_<Point3f> numerator;
		Mat_<float> numerator_single_channel;
        Mat_<float> denominator;
        Mat_<Point3f> numerator_filtered;
		Mat_<float> numerator_filtered_single_channel;
        Mat_<float> denominator_filtered;

        Mat_<Point3f> X;
		Mat_<float> X_single_channel;
        Mat_<Point3f> eta_k_small;
		Mat_<float> eta_k_small_single_channel;
        Mat_<Point3f> eta_k_big;
		Mat_<float> eta_k_big_single_channel;
        Mat_<Point3f> X_squared;
		Mat_<float> X_squared_single_channel;
        Mat_<float> pixel_dist_to_manifold_squared;
        Mat_<float> gaussian_distance_weights;
        Mat_<Point3f> Psi_splat;
		Mat_<float> Psi_splat_single_channel;
        Mat_<Vec4f> Psi_splat_joined;
		Mat_<Vec2f> Psi_splat_joined_single_channel;
        Mat_<Vec4f> Psi_splat_joined_resized;
		Mat_<Vec2f> Psi_splat_joined_resized_single_channel;
        Mat_<Vec4f> blurred_projected_values;
		Mat_<Vec2f> blurred_projected_values_single_channel;
        Mat_<Point3f> w_ki_Psi_blur;
		Mat_<float> w_ki_Psi_blur_single_channel;
        Mat_<float> w_ki_Psi_blur_0;
        Mat_<Point3f> w_ki_Psi_blur_resized;
		Mat_<float> w_ki_Psi_blur_resized_single_channel;
        Mat_<float> w_ki_Psi_blur_0_resized;
        Mat_<float> rand_vec;
        Mat_<float> v1;
        Mat_<float> Nx_v1_mult;
        Mat_<float> theta;

        std::vector<Mat_<Point3f> > eta_minus;
		std::vector<Mat_<float> > eta_minus_single_channel;
        std::vector<Mat_<uchar> > cluster_minus;
        std::vector<Mat_<Point3f> > eta_plus;
		std::vector<Mat_<float> > eta_plus_single_channel;
        std::vector<Mat_<uchar> > cluster_plus;

        void release();
    };

    void Buf::release()
    {
        eta_1.release();
		eta_1_single_channel.release();
        cluster_1.release();

        tilde_dst.release();
		tilde_dst_single_channel.release();
        alpha.release();
        diff.release();
		diff_single_channel.release();
        dst.release();
		dst_single_channel.release();

        V.release();

        dIcdx.release();
		dIcdx_single_channel.release();
        dIcdy.release();
		dIcdy_single_channel.release();
        dIdx.release();
        dIdy.release();
        dHdx.release();
        dVdy.release();

        t.release();

        theta_masked.release();
        mul.release();
		mul_single_channel.release();
        numerator.release();
		numerator_single_channel.release();
        denominator.release();
        numerator_filtered.release();
		numerator_filtered_single_channel.release();
        denominator_filtered.release();

        X.release();
		X_single_channel.release();
        eta_k_small.release();
		eta_k_small_single_channel.release();
        eta_k_big.release();
		eta_k_big_single_channel.release();
        X_squared.release();
		X_squared_single_channel.release();
        pixel_dist_to_manifold_squared.release();
        gaussian_distance_weights.release();
        Psi_splat.release();
		Psi_splat_single_channel.release();
        Psi_splat_joined.release();
		Psi_splat_joined_single_channel.release();
        Psi_splat_joined_resized.release();
		Psi_splat_joined_resized_single_channel.release();
        blurred_projected_values.release();
		blurred_projected_values_single_channel.release();
        w_ki_Psi_blur.release();
		w_ki_Psi_blur_single_channel.release();
        w_ki_Psi_blur_0.release();
        w_ki_Psi_blur_resized.release();
		w_ki_Psi_blur_resized_single_channel.release();
        w_ki_Psi_blur_0_resized.release();
        rand_vec.release();
        v1.release();
        Nx_v1_mult.release();
        theta.release();

        eta_minus.clear();
		eta_minus_single_channel.clear();
        cluster_minus.clear();
        eta_plus.clear();
		eta_plus_single_channel.clear();
        cluster_plus.clear();
    }
    
	// declare the class to implement adaptive manifold filter
    class AdaptiveManifoldFilterImpl : public AdaptiveManifoldFilter
    {
    public:
        AlgorithmInfo* info() const;

        AdaptiveManifoldFilterImpl();

        void apply(InputArray src, OutputArray dst, OutputArray tilde_dst = noArray(), InputArray src_joint = noArray());

		void apply_on_cv32fc1(InputArray _src, OutputArray _dst, OutputArray _tilde_dst = noArray(), InputArray _src_joint = noArray());

        void collectGarbage();

    protected:
        double sigma_s_;
        double sigma_r_;
        int tree_height_;
        int num_pca_iterations_;

    private:
        void buildManifoldsAndPerformFiltering(const Mat_<Point3f>& eta_k, const Mat_<uchar>& cluster_k, int current_tree_level);
		void buildManifoldsAndPerformFiltering_single_channel(const Mat_<float>& eta_k, const Mat_<uchar>& cluster_k, int current_tree_level);

        Buf buf_;

        Mat_<Point3f> src_f_;
        Mat_<Point3f> src_joint_f_;

		Mat_<float> src_f_single_channel;
		Mat_<float> src_joint_f_single_channel;

        Mat_<Point3f> sum_w_ki_Psi_blur_;
		Mat_<float> sum_w_ki_Psi_blur_single_channel;
        Mat_<float> sum_w_ki_Psi_blur_0_;

        Mat_<float> min_pixel_dist_to_manifold_squared_;

        RNG rng_;

        int cur_tree_height_;
        float sigma_r_over_sqrt_2_;
    };

	// constructor 
    AdaptiveManifoldFilterImpl::AdaptiveManifoldFilterImpl()
    {
        sigma_s_ = 16.0;
        sigma_r_ = 0.2;
        tree_height_ = -1;
        num_pca_iterations_ = 1;
    }

    void AdaptiveManifoldFilterImpl::collectGarbage()
    {
        buf_.release();

        src_f_.release();
        src_joint_f_.release();

        sum_w_ki_Psi_blur_.release();
		sum_w_ki_Psi_blur_single_channel.release();
        sum_w_ki_Psi_blur_0_.release();

        min_pixel_dist_to_manifold_squared_.release();
    }

    CV_INIT_ALGORITHM(AdaptiveManifoldFilterImpl, "AdaptiveManifoldFilter",
                      obj.info()->addParam(obj, "sigma_s", obj.sigma_s_, false, 0, 0, "Filter spatial standard deviation");
                      obj.info()->addParam(obj, "sigma_r", obj.sigma_r_, false, 0, 0, "Filter range standard deviation");
                      obj.info()->addParam(obj, "tree_height", obj.tree_height_, false, 0, 0, "Height of the manifold tree (default = -1 : automatically computed)");
                      obj.info()->addParam(obj, "num_pca_iterations", obj.num_pca_iterations_, false, 0, 0, "Number of iterations to computed the eigenvector v1"));

    // calculate the logarithm
	inline double Log2(double n)
    {
        return log(n) / log(2.0);
    }

    // compute the tree height in adaptive manifold filter
	inline int computeManifoldTreeHeight(double sigma_s, double sigma_r)
    {
        const double Hs = floor(Log2(sigma_s)) - 1.0;
        const double Lr = 1.0 - sigma_r;
        return max(2, static_cast<int>(ceil(Hs * Lr)));
    }

    // ensures that the size of a matrix is big enough and the matrix has a proper type
    void ensureSizeIsEnough(int rows, int cols, int type, Mat& m)
    {
        if (m.empty() || m.type() != type || m.data != m.datastart)
            m.create(rows, cols, type);
        else
        {
            const size_t esz = m.elemSize();   // channels
            const ptrdiff_t delta2 = m.dataend - m.datastart;

            const size_t minstep = m.cols * esz;

            Size wholeSize;
            wholeSize.height = std::max(static_cast<int>((delta2 - minstep) / m.step /* bytes of row */ + 1), m.rows);
            wholeSize.width = std::max(static_cast<int>((delta2 - m.step * (wholeSize.height - 1)) / esz), m.cols);

            if (wholeSize.height < rows || wholeSize.width < cols)
                m.create(rows, cols, type);
            else
            {
                m.cols = cols;
                m.rows = rows;
            }
        }
    }

    inline void ensureSizeIsEnough(Size size, int type, Mat& m)
    {
        ensureSizeIsEnough(size.height, size.width, type, m);
    }

    template <typename T>
    inline void ensureSizeIsEnough(int rows, int cols, Mat_<T>& m)
    {
        if (m.empty() || m.data != m.datastart)
            m.create(rows, cols);
        else
        {
            const size_t esz = m.elemSize();
            const ptrdiff_t delta2 = m.dataend - m.datastart;

            const size_t minstep = m.cols * esz;

            Size wholeSize;
            wholeSize.height = std::max(static_cast<int>((delta2 - minstep) / m.step + 1), m.rows);
            wholeSize.width = std::max(static_cast<int>((delta2 - m.step * (wholeSize.height - 1)) / esz), m.cols);

            if (wholeSize.height < rows || wholeSize.width < cols)
                m.create(rows, cols);
            else
            {
                m.cols = cols;
                m.rows = rows;
            }
        }
    }

    template <typename T>
    inline void ensureSizeIsEnough(Size size, Mat_<T>& m)
    {
        ensureSizeIsEnough(size.height, size.width, m);
    }

    // filter aiming at mat which belongs to CV_32F
	template <typename T>
    void h_filter(const Mat_<T>& src, Mat_<T>& dst, float sigma)
    {
        CV_DbgAssert( src.depth() == CV_32F );

        const float a = exp(-sqrt(2.0f) / sigma);

        ensureSizeIsEnough(src.size(), dst);
        src.copyTo(dst);

        for (int y = 0; y < src.rows; ++y)
        {
            const T* src_row = src[y];
            T* dst_row = dst[y];

            for (int x = 1; x < src.cols; ++x)
            {
                dst_row[x] = src_row[x] + a * (src_row[x - 1] - src_row[x]);
            }
            for (int x = src.cols - 2; x >= 0; --x)
            {
                dst_row[x] = dst_row[x] + a * (dst_row[x + 1] - dst_row[x]);
            }
        }

        for (int y = 1; y < src.rows; ++y)
        {
            T* dst_cur_row = dst[y];
            T* dst_prev_row = dst[y - 1];

            for (int x = 0; x < src.cols; ++x)
            {
                dst_cur_row[x] = dst_cur_row[x] + a * (dst_prev_row[x] - dst_cur_row[x]);
            }
        }
        for (int y = src.rows - 2; y >= 0; --y)
        {
            T* dst_cur_row = dst[y];
            T* dst_prev_row = dst[y + 1];

            for (int x = 0; x < src.cols; ++x)
            {
                dst_cur_row[x] = dst_cur_row[x] + a * (dst_prev_row[x] - dst_cur_row[x]);
            }
        }
    }

    // divide 
	template <typename T>
    void rdivide(const Mat_<T>& a, const Mat_<float>& b, Mat_<T>& dst)
    {
        CV_DbgAssert( a.depth() == CV_32F );
        CV_DbgAssert( a.size() == b.size() );

        ensureSizeIsEnough(a.size(), dst);
        dst.setTo(0);

        for (int y = 0; y < a.rows; ++y)
        {
            const T* a_row = a[y];
            const float* b_row = b[y];
            T* dst_row = dst[y];

            for (int x = 0; x < a.cols; ++x)
            {
                if (b_row[x] > numeric_limits<float>::epsilon())
                    dst_row[x] = a_row[x] * (1.0f / b_row[x]);
            }
        }
    }

    // multiply
	template <typename T>
    void times(const Mat_<T>& a, const Mat_<float>& b, Mat_<T>& dst)
    {
        CV_DbgAssert( a.depth() == CV_32F );
        CV_DbgAssert( a.size() == b.size() );

        ensureSizeIsEnough(a.size(), dst);

        for (int y = 0; y < a.rows; ++y)
        {
            const T* a_row = a[y];
            const float* b_row = b[y];
            T* dst_row = dst[y];

            for (int x = 0; x < a.cols; ++x)
            {
                dst_row[x] = a_row[x] * b_row[x];
            }
        }
    }

    void AdaptiveManifoldFilterImpl::apply(InputArray _src, OutputArray _dst, OutputArray _tilde_dst, InputArray _src_joint)
    {
        const Mat src = _src.getMat();
        const Mat src_joint = _src_joint.getMat();

        const Size srcSize = src.size();

        CV_Assert( src.type() == CV_8UC3);
        CV_Assert( src_joint.empty() || (src_joint.type() == src.type() && src_joint.size() == srcSize));

        ensureSizeIsEnough(srcSize, src_f_);
        src.convertTo(src_f_, src_f_.type(), 1.0 / 255.0);

        // Use the center pixel as seed to random number generation.
        const Point3f centralPix = src_f_(src_f_.rows / 2, src_f_.cols / 2);
        const double seedCoeff = (centralPix.x + centralPix.y + centralPix.z + 1.0f) / 4.0f;
        rng_.state = static_cast<uint64>(seedCoeff * numeric_limits<uint64>::max());

        ensureSizeIsEnough(srcSize, sum_w_ki_Psi_blur_);
        sum_w_ki_Psi_blur_.setTo(Scalar::all(0));

        ensureSizeIsEnough(srcSize, sum_w_ki_Psi_blur_0_);
        sum_w_ki_Psi_blur_0_.setTo(Scalar::all(0));

        ensureSizeIsEnough(srcSize, min_pixel_dist_to_manifold_squared_);
        min_pixel_dist_to_manifold_squared_.setTo(Scalar::all(numeric_limits<float>::max()));

        // If the tree_height was not specified, compute it using Eq. (10) of our paper.
        cur_tree_height_ = tree_height_ > 0 ? tree_height_ : computeManifoldTreeHeight(sigma_s_, sigma_r_);

        // If no joint signal was specified, use the original signal
        ensureSizeIsEnough(srcSize, src_joint_f_);
        if (src_joint.empty())
            src_f_.copyTo(src_joint_f_);
        else
            src_joint.convertTo(src_joint_f_, src_joint_f_.type(), 1.0 / 255.0);

        // Dividing the covariance matrix by 2 is equivalent to dividing the standard deviations by sqrt(2).
        sigma_r_over_sqrt_2_ = static_cast<float>(sigma_r_ / sqrt(2.0));

        // Algorithm 1, Step 1: compute the first manifold by low-pass filtering.
        h_filter(src_joint_f_, buf_.eta_1, static_cast<float>(sigma_s_));

        ensureSizeIsEnough(srcSize, buf_.cluster_1);
        buf_.cluster_1.setTo(Scalar::all(1));

        buf_.eta_minus.resize(cur_tree_height_);
        buf_.cluster_minus.resize(cur_tree_height_);
        buf_.eta_plus.resize(cur_tree_height_);
        buf_.cluster_plus.resize(cur_tree_height_);
        buildManifoldsAndPerformFiltering(buf_.eta_1, buf_.cluster_1, 1);

        // Compute the filter response by normalized convolution -- Eq. (4)
        rdivide(sum_w_ki_Psi_blur_, sum_w_ki_Psi_blur_0_, buf_.tilde_dst);

        // Adjust the filter response for outlier pixels -- Eq. (10)
        ensureSizeIsEnough(srcSize, buf_.alpha);
        exp(min_pixel_dist_to_manifold_squared_ * (-0.5 / sigma_r_ / sigma_r_), buf_.alpha);

        ensureSizeIsEnough(srcSize, buf_.diff);
        subtract(buf_.tilde_dst, src_f_, buf_.diff);
        times(buf_.diff, buf_.alpha, buf_.diff);

        ensureSizeIsEnough(srcSize, buf_.dst);
        add(src_f_, buf_.diff, buf_.dst);

        buf_.dst.convertTo(_dst, CV_8U, 255.0);
        if (_tilde_dst.needed())
            buf_.tilde_dst.convertTo(_tilde_dst, CV_8U, 255.0);
    }

	void AdaptiveManifoldFilterImpl::apply_on_cv32fc1(InputArray _src, OutputArray _dst, OutputArray _tilde_dst, InputArray _src_joint)
	{
		const Mat src = _src.getMat();
		const Mat src_joint = _src_joint.getMat();

		const Size srcSize = src.size();

		CV_Assert( src.type() == CV_32FC1);
		CV_Assert( src_joint.empty() || (src_joint.type() == src.type() && src_joint.size() == srcSize));

		ensureSizeIsEnough(srcSize, src_f_single_channel);
	//	src.convertTo(src_f_single_channel, src_f_single_channel.type(), 1.0 / 255.0);
		src.copyTo (src_f_single_channel);

		// Use the center pixel as seed to random number generation.
		const float centralPix = src_f_single_channel(src_f_single_channel.rows / 2, src_f_single_channel.cols / 2);
		/*const double seedCoeff = (centralPix + 1.0f) / 2.0f;*/ 
		const double seedCoeff = (centralPix + 1.0f) / 2.0f;
		rng_.state = static_cast<uint64>(seedCoeff * numeric_limits<uint64>::max());

		ensureSizeIsEnough(srcSize, sum_w_ki_Psi_blur_single_channel);
		sum_w_ki_Psi_blur_single_channel.setTo(Scalar::all(0));

		ensureSizeIsEnough(srcSize, sum_w_ki_Psi_blur_0_);
		sum_w_ki_Psi_blur_0_.setTo(Scalar::all(0));

		ensureSizeIsEnough(srcSize, min_pixel_dist_to_manifold_squared_);
		min_pixel_dist_to_manifold_squared_.setTo(Scalar::all(numeric_limits<float>::max()));

		// If the tree_height was not specified, compute it using Eq. (10) of our paper.
		cur_tree_height_ = tree_height_ > 0 ? tree_height_ : computeManifoldTreeHeight(sigma_s_, sigma_r_);

		// If no joint signal was specified, use the original signal
		ensureSizeIsEnough(srcSize, src_joint_f_single_channel);
		if (src_joint.empty())
			src_f_single_channel.copyTo(src_joint_f_single_channel);
		else
//			src_joint.convertTo(src_joint_f_single_channel, src_joint_f_single_channel.type(), 1.0 / 255.0);
            src_joint.copyTo(src_joint_f_single_channel);

		// Dividing the covariance matrix by 2 is equivalent to dividing the standard deviations by sqrt(2).
		sigma_r_over_sqrt_2_ = static_cast<float>(sigma_r_ / sqrt(2.0));

		// Algorithm 1, Step 1: compute the first manifold by low-pass filtering.
		h_filter(src_joint_f_single_channel, buf_.eta_1_single_channel, static_cast<float>(sigma_s_));

		ensureSizeIsEnough(srcSize, buf_.cluster_1);
		buf_.cluster_1.setTo(Scalar::all(1));

		buf_.eta_minus_single_channel.resize(cur_tree_height_);
		buf_.cluster_minus.resize(cur_tree_height_);
		buf_.eta_plus_single_channel.resize(cur_tree_height_);
		buf_.cluster_plus.resize(cur_tree_height_);
		buildManifoldsAndPerformFiltering_single_channel(buf_.eta_1_single_channel, buf_.cluster_1, 1);

		// Compute the filter response by normalized convolution -- Eq. (4)
		rdivide(sum_w_ki_Psi_blur_single_channel, sum_w_ki_Psi_blur_0_, buf_.tilde_dst_single_channel);

		// Adjust the filter response for outlier pixels -- Eq. (10)
		ensureSizeIsEnough(srcSize, buf_.alpha);
		exp(min_pixel_dist_to_manifold_squared_ * (-0.5 / sigma_r_ / sigma_r_), buf_.alpha);

		ensureSizeIsEnough(srcSize, buf_.diff_single_channel);
		subtract(buf_.tilde_dst_single_channel, src_f_single_channel, buf_.diff_single_channel);
		times(buf_.diff_single_channel, buf_.alpha, buf_.diff_single_channel);

		ensureSizeIsEnough(srcSize, buf_.diff_single_channel);
		add(src_f_single_channel, buf_.diff_single_channel, buf_.diff_single_channel);

//		buf_.dst_single_channel.convertTo(_dst, CV_32F, 255.0);
		buf_.dst_single_channel.copyTo(_dst);
		if (_tilde_dst.needed())
//			buf_.tilde_dst_single_channel.convertTo(_tilde_dst, CV_32F, 255.0);
            buf_.tilde_dst_single_channel.copyTo(_tilde_dst);
	}

	// floor to power of two
    inline double floor_to_power_of_two(double r)
    {
        return pow(2.0, floor(Log2(r)));
    }

    // get the sum of data belongs to different channels
    void channelsSum(const Mat_<Point3f>& src, Mat_<float>& dst)
    {
        ensureSizeIsEnough(src.size(), dst);

        for (int y = 0; y < src.rows; ++y)
        {
            const Point3f* src_row = src[y];
            float* dst_row = dst[y];

            for (int x = 0; x < src.cols; ++x)
            {
                const Point3f src_val = src_row[x];
                dst_row[x] = src_val.x + src_val.y + src_val.z;
            }
        }
    }

    // phi
	void phi(const Mat_<float>& src, Mat_<float>& dst, float sigma)
    {
        ensureSizeIsEnough(src.size(), dst);

        for (int y = 0; y < dst.rows; ++y)
        {
            const float* src_row = src[y];
            float* dst_row = dst[y];

            for (int x = 0; x < dst.cols; ++x)
            {
                dst_row[x] = exp(-0.5f * src_row[x] / sigma / sigma);
            }
        }
    }

	// put data with three and one channel together
    void catCn(const Mat_<Point3f>& a, const Mat_<float>& b, Mat_<Vec4f>& dst)
    {
        ensureSizeIsEnough(a.size(), dst);

        for (int y = 0; y < a.rows; ++y)
        {
            const Point3f* a_row = a[y];
            const float* b_row = b[y];
            Vec4f* dst_row = dst[y];

            for (int x = 0; x < a.cols; ++x)
            {
                const Point3f a_val = a_row[x];
                const float b_val = b_row[x];
                dst_row[x] = Vec4f(a_val.x, a_val.y, a_val.z, b_val);
            }
        }
    }

	void catCn2Channels(const Mat_<float>& a, const Mat_<float>& b, Mat_<Vec2f>& dst)
	{
		ensureSizeIsEnough(a.size(), dst);

		for (int y = 0; y < a.rows; ++y)
		{
			const float* a_row = a[y];
			const float* b_row = b[y];
			Vec2f* dst_row = dst[y];

			for (int x = 0; x < a.cols; ++x)
			{
				const float a_val = a_row[x];
				const float b_val = b_row[x];
				dst_row[x] = Vec2f(a_val, b_val);
			}
		}
	}

    // get the difference from the Y direction
	void diffY(const Mat_<Point3f>& src, Mat_<Point3f>& dst)
    {
        ensureSizeIsEnough(src.rows - 1, src.cols, dst);

        for (int y = 0; y < src.rows - 1; ++y)
        {
            const Point3f* src_cur_row = src[y];
            const Point3f* src_next_row = src[y + 1];
            Point3f* dst_row = dst[y];

            for (int x = 0; x < src.cols; ++x)
            {
                dst_row[x] = src_next_row[x] - src_cur_row[x];
            }
        }
    }

	void diffY_single_channel(const Mat_<float>& src, Mat_<float>& dst)
	{
		ensureSizeIsEnough(src.rows - 1, src.cols, dst);

		for (int y = 0; y < src.rows - 1; ++y)
		{
			const float* src_cur_row = src[y];
			const float* src_next_row = src[y + 1];
			float* dst_row = dst[y];

			for (int x = 0; x < src.cols; ++x)
			{
				dst_row[x] = src_next_row[x] - src_cur_row[x];
			}
		}
	}

    // get the difference from the X direction
	void diffX(const Mat_<Point3f>& src, Mat_<Point3f>& dst)
    {
        ensureSizeIsEnough(src.rows, src.cols - 1, dst);

        for (int y = 0; y < src.rows; ++y)
        {
            const Point3f* src_row = src[y];
            Point3f* dst_row = dst[y];

            for (int x = 0; x < src.cols - 1; ++x)
            {
                dst_row[x] = src_row[x + 1] - src_row[x];
            }
        }
    }

	void diffX_single_channel(const Mat_<float>& src, Mat_<float>& dst)
	{
		ensureSizeIsEnough(src.rows, src.cols - 1, dst);

		for (int y = 0; y < src.rows; ++y)
		{
			const float* src_row = src[y];
			float* dst_row = dst[y];

			for (int x = 0; x < src.cols - 1; ++x)
			{
				dst_row[x] = src_row[x + 1] - src_row[x];
			}
		}
	}

    void TransformedDomainRecursiveFilter(const Mat_<Vec4f>& I, const Mat_<float>& DH, const Mat_<float>& DV, Mat_<Vec4f>& dst, float sigma, Buf& buf)
    {
        CV_DbgAssert( I.size() == DH.size() );
		
        const float a = exp(-sqrt(2.0f) / sigma);

        ensureSizeIsEnough(I.size(), dst);
        I.copyTo(dst);

        ensureSizeIsEnough(DH.size(), buf.V);

        for (int y = 0; y < DH.rows; ++y)
        {
            const float* D_row = DH[y];
            float* V_row = buf.V[y];

            for (int x = 0; x < DH.cols; ++x)
            {
                V_row[x] = pow(a, D_row[x]);
            }
        }
        for (int y = 0; y < I.rows; ++y)
        {
            const float* V_row = buf.V[y];
            Vec4f* dst_row = dst[y];

            for (int x = 1; x < I.cols; ++x)
            {
                Vec4f dst_cur_val = dst_row[x];
                const Vec4f dst_prev_val = dst_row[x - 1];
                const float V_val = V_row[x];

                dst_cur_val[0] += V_val * (dst_prev_val[0] - dst_cur_val[0]);
                dst_cur_val[1] += V_val * (dst_prev_val[1] - dst_cur_val[1]);
                dst_cur_val[2] += V_val * (dst_prev_val[2] - dst_cur_val[2]);
                dst_cur_val[3] += V_val * (dst_prev_val[3] - dst_cur_val[3]);

                dst_row[x] = dst_cur_val;
            }
            for (int x = I.cols - 2; x >= 0; --x)
            {
                Vec4f dst_cur_val = dst_row[x];
                const Vec4f dst_prev_val = dst_row[x + 1];
                const float V_val = V_row[x];

                dst_cur_val[0] += V_val * (dst_prev_val[0] - dst_cur_val[0]);
                dst_cur_val[1] += V_val * (dst_prev_val[1] - dst_cur_val[1]);
                dst_cur_val[2] += V_val * (dst_prev_val[2] - dst_cur_val[2]);
                dst_cur_val[3] += V_val * (dst_prev_val[3] - dst_cur_val[3]);

                dst_row[x] = dst_cur_val;
            }
        }

        for (int y = 0; y < DV.rows; ++y)
        {
            const float* D_row = DV[y];
            float* V_row = buf.V[y];

            for (int x = 0; x < DV.cols; ++x)
            {
                V_row[x] = pow(a, D_row[x]);
            }
        }
        for (int y = 1; y < I.rows; ++y)
        {
            const float* V_row = buf.V[y];
            Vec4f* dst_cur_row = dst[y];
            Vec4f* dst_prev_row = dst[y - 1];

            for (int x = 0; x < I.cols; ++x)
            {
                Vec4f dst_cur_val = dst_cur_row[x];
                const Vec4f dst_prev_val = dst_prev_row[x];
                const float V_val = V_row[x];

                dst_cur_val[0] += V_val * (dst_prev_val[0] - dst_cur_val[0]);
                dst_cur_val[1] += V_val * (dst_prev_val[1] - dst_cur_val[1]);
                dst_cur_val[2] += V_val * (dst_prev_val[2] - dst_cur_val[2]);
                dst_cur_val[3] += V_val * (dst_prev_val[3] - dst_cur_val[3]);

                dst_cur_row[x] = dst_cur_val;
            }
        }
        for (int y = I.rows - 2; y >= 0; --y)
        {
            const float* V_row = buf.V[y];
            Vec4f* dst_cur_row = dst[y];
            Vec4f* dst_prev_row = dst[y + 1];

            for (int x = 0; x < I.cols; ++x)
            {
                Vec4f dst_cur_val = dst_cur_row[x];
                const Vec4f dst_prev_val = dst_prev_row[x];
                const float V_val = V_row[x];

                dst_cur_val[0] += V_val * (dst_prev_val[0] - dst_cur_val[0]);
                dst_cur_val[1] += V_val * (dst_prev_val[1] - dst_cur_val[1]);
                dst_cur_val[2] += V_val * (dst_prev_val[2] - dst_cur_val[2]);
                dst_cur_val[3] += V_val * (dst_prev_val[3] - dst_cur_val[3]);

                dst_cur_row[x] = dst_cur_val;
            }
        }
    }

    void TransformedDomainRecursiveFilter_single_channel(const Mat_<Vec2f>& I, const Mat_<float>& DH, const Mat_<float>& DV, Mat_<Vec2f>& dst, float sigma, Buf& buf)
	{
		CV_DbgAssert( I.size() == DH.size() );

		const float a = exp(-sqrt(2.0f) / sigma);

		ensureSizeIsEnough(I.size(), dst);
		I.copyTo(dst);

		ensureSizeIsEnough(DH.size(), buf.V);

		for (int y = 0; y < DH.rows; ++y)
		{
			const float* D_row = DH[y];
			float* V_row = buf.V[y];

			for (int x = 0; x < DH.cols; ++x)
			{
				V_row[x] = pow(a, D_row[x]);
			}
		}
		for (int y = 0; y < I.rows; ++y)
		{
			const float* V_row = buf.V[y];
			Vec2f* dst_row = dst[y];

			for (int x = 1; x < I.cols; ++x)
			{
				Vec2f dst_cur_val = dst_row[x];
				const Vec2f dst_prev_val = dst_row[x - 1];
				const float V_val = V_row[x];

				dst_cur_val[0] += V_val * (dst_prev_val[0] - dst_cur_val[0]);
				dst_cur_val[1] += V_val * (dst_prev_val[1] - dst_cur_val[1]);

				dst_row[x] = dst_cur_val;
			}
			for (int x = I.cols - 2; x >= 0; --x)
			{
				Vec2f dst_cur_val = dst_row[x];
				const Vec2f dst_prev_val = dst_row[x + 1];
				const float V_val = V_row[x];

				dst_cur_val[0] += V_val * (dst_prev_val[0] - dst_cur_val[0]);
				dst_cur_val[1] += V_val * (dst_prev_val[1] - dst_cur_val[1]);

				dst_row[x] = dst_cur_val;
			}
		}

		for (int y = 0; y < DV.rows; ++y)
		{
			const float* D_row = DV[y];
			float* V_row = buf.V[y];

			for (int x = 0; x < DV.cols; ++x)
			{
				V_row[x] = pow(a, D_row[x]);
			}
		}
		for (int y = 1; y < I.rows; ++y)
		{
			const float* V_row = buf.V[y];
			Vec2f* dst_cur_row = dst[y];
			Vec2f* dst_prev_row = dst[y - 1];

			for (int x = 0; x < I.cols; ++x)
			{
				Vec2f dst_cur_val = dst_cur_row[x];
				const Vec2f dst_prev_val = dst_prev_row[x];
				const float V_val = V_row[x];

				dst_cur_val[0] += V_val * (dst_prev_val[0] - dst_cur_val[0]);
				dst_cur_val[1] += V_val * (dst_prev_val[1] - dst_cur_val[1]);

				dst_cur_row[x] = dst_cur_val;
			}
		}
		for (int y = I.rows - 2; y >= 0; --y)
		{
			const float* V_row = buf.V[y];
			Vec2f* dst_cur_row = dst[y];
			Vec2f* dst_prev_row = dst[y + 1];

			for (int x = 0; x < I.cols; ++x)
			{
				Vec2f dst_cur_val = dst_cur_row[x];
				const Vec2f dst_prev_val = dst_prev_row[x];
				const float V_val = V_row[x];

				dst_cur_val[0] += V_val * (dst_prev_val[0] - dst_cur_val[0]);
				dst_cur_val[1] += V_val * (dst_prev_val[1] - dst_cur_val[1]);

				dst_cur_row[x] = dst_cur_val;
			}
		}
	}
	
	void RF_filter(const Mat_<Vec4f>& src, const Mat_<Point3f>& src_joint, Mat_<Vec4f>& dst, float sigma_s, float sigma_r, Buf& buf)
    {
        CV_DbgAssert( src_joint.size() == src.size() );

        diffX(src_joint, buf.dIcdx);
        diffY(src_joint, buf.dIcdy);

        ensureSizeIsEnough(src.size(), buf.dIdx);
        buf.dIdx.setTo(Scalar::all(0));
        for (int y = 0; y < src.rows; ++y)
        {
            const Point3f* dIcdx_row = buf.dIcdx[y];
            float* dIdx_row = buf.dIdx[y];

            for (int x = 1; x < src.cols; ++x)
            {
                const Point3f val = dIcdx_row[x - 1];
                dIdx_row[x] = val.dot(val);
            }
        }

        ensureSizeIsEnough(src.size(), buf.dIdy);
        buf.dIdy.setTo(Scalar::all(0));
        for (int y = 1; y < src.rows; ++y)
        {
            const Point3f* dIcdy_row = buf.dIcdy[y - 1];
            float* dIdy_row = buf.dIdy[y];

            for (int x = 0; x < src.cols; ++x)
            {
                const Point3f val = dIcdy_row[x];
                dIdy_row[x] = val.dot(val);
            }
        }

        ensureSizeIsEnough(buf.dIdx.size(), buf.dHdx);
        buf.dIdx.convertTo(buf.dHdx, buf.dHdx.type(), (sigma_s / sigma_r) * (sigma_s / sigma_r), (sigma_s / sigma_s) * (sigma_s / sigma_s));
        sqrt(buf.dHdx, buf.dHdx);

        ensureSizeIsEnough(buf.dIdy.size(), buf.dVdy);
        buf.dIdy.convertTo(buf.dVdy, buf.dVdy.type(), (sigma_s / sigma_r) * (sigma_s / sigma_r), (sigma_s / sigma_s) * (sigma_s / sigma_s));
        sqrt(buf.dVdy, buf.dVdy);

        ensureSizeIsEnough(src.size(), dst);
        src.copyTo(dst);
        TransformedDomainRecursiveFilter(src, buf.dHdx, buf.dVdy, dst, sigma_s, buf);
    }

    void RF_filter_single_channel(const Mat_<Vec2f>& src, const Mat_<float>& src_joint, Mat_<Vec2f>& dst, float sigma_s, float sigma_r, Buf& buf)
	{
		CV_DbgAssert( src_joint.size() == src.size());

		diffX_single_channel(src_joint, buf.dIcdx_single_channel);
		diffY_single_channel(src_joint, buf.dIcdy_single_channel);

		ensureSizeIsEnough(src.size(), buf.dIdx);
		buf.dIdx.setTo(Scalar::all(0));
		for (int y = 0; y < src.rows; ++y)
		{
			const float* dIcdx_row = buf.dIcdx_single_channel[y];
			float* dIdx_row = buf.dIdx[y];

			for (int x = 1; x < src.cols; ++x)
			{
				const float val = dIcdx_row[x - 1];
				dIdx_row[x] = val * val;    // ??
			}
		}

		ensureSizeIsEnough(src.size(), buf.dIdy);
		buf.dIdy.setTo(Scalar::all(0));
		for (int y = 1; y < src.rows; ++y)
		{
			const float* dIcdy_row = buf.dIcdy_single_channel[y - 1];
			float* dIdy_row = buf.dIdy[y];

			for (int x = 0; x < src.cols; ++x)
			{
				const float val = dIcdy_row[x];
				dIdy_row[x] = val * val;    // ??
			}
		}

		ensureSizeIsEnough(buf.dIdx.size(), buf.dHdx);
		buf.dIdx.convertTo(buf.dHdx, buf.dHdx.type(), (sigma_s / sigma_r) * (sigma_s / sigma_r), (sigma_s / sigma_s) * (sigma_s / sigma_s));
		sqrt(buf.dHdx, buf.dHdx);

		ensureSizeIsEnough(buf.dIdy.size(), buf.dVdy);
		buf.dIdy.convertTo(buf.dVdy, buf.dVdy.type(), (sigma_s / sigma_r) * (sigma_s / sigma_r), (sigma_s / sigma_s) * (sigma_s / sigma_s));
		sqrt(buf.dVdy, buf.dVdy);

		ensureSizeIsEnough(src.size(), dst);
		src.copyTo(dst);
		TransformedDomainRecursiveFilter_single_channel(src, buf.dHdx, buf.dVdy, dst, sigma_s, buf);
	}
	
	// split data from four channels into three channels and one channel
	void split_3_1(const Mat_<Vec4f>& src, Mat_<Point3f>& dst1, Mat_<float>& dst2)
    {
        ensureSizeIsEnough(src.size(), dst1);
        ensureSizeIsEnough(src.size(), dst2);

        for (int y = 0; y < src.rows; ++y)
        {
            const Vec4f* src_row = src[y];
            Point3f* dst1_row = dst1[y];
            float* dst2_row = dst2[y];

            for (int x = 0; x < src.cols; ++x)
            {
                Vec4f val = src_row[x];
                dst1_row[x] = Point3f(val[0], val[1], val[2]);
                dst2_row[x] = val[3];
            }
        }
    }

	void split_1_1(const Mat_<Vec2f>& src, Mat_<float>& dst1, Mat_<float>& dst2)
	{
		ensureSizeIsEnough(src.size(), dst1);
		ensureSizeIsEnough(src.size(), dst2);

		for (int y = 0; y < src.rows; ++y)
		{
			const Vec2f* src_row = src[y];
			float* dst1_row = dst1[y];
			float* dst2_row = dst2[y];

			for (int x = 0; x < src.cols; ++x)
			{
				Vec2f val = src_row[x];
				dst1_row[x] = float(val[0]);
				dst2_row[x] = val[1];
			}
		}
	}

    void computeEigenVector(const Mat_<float>& X, const Mat_<uchar>& mask, Mat_<float>& dst, int num_pca_iterations, const Mat_<float>& rand_vec, Buf& buf)
    {
        CV_DbgAssert( X.cols == rand_vec.cols );
        CV_DbgAssert( X.rows == mask.size().area() );
        CV_DbgAssert( rand_vec.rows == 1 );

        ensureSizeIsEnough(rand_vec.size(), dst);
        rand_vec.copyTo(dst);

        ensureSizeIsEnough(X.size(), buf.t);

        float* dst_row = dst[0];

        for (int i = 0; i < num_pca_iterations; ++i)
        {
            buf.t.setTo(Scalar::all(0));

            for (int y = 0, ind = 0; y < mask.rows; ++y)
            {
                const uchar* mask_row = mask[y];

                for (int x = 0; x < mask.cols; ++x, ++ind)
                {
                    if (mask_row[x])
                    {
                        const float* X_row = X[ind];
                        float* t_row = buf.t[ind];

                        float dots = 0.0;
                        for (int c = 0; c < X.cols; ++c)
                            dots += dst_row[c] * X_row[c];

                        for (int c = 0; c < X.cols; ++c)
                            t_row[c] = dots * X_row[c];
                    }
                }
            }

            dst.setTo(0.0);
            for (int i = 0; i < X.rows; ++i)
            {
                const float* t_row = buf.t[i];

                for (int c = 0; c < X.cols; ++c)
                {
                    dst_row[c] += t_row[c];
                }
            }
        }

        double n = norm(dst);
        divide(dst, n, dst);
    }

    void calcEta(const Mat_<Point3f>& src_joint_f, const Mat_<float>& theta, const Mat_<uchar>& cluster, Mat_<Point3f>& dst, float sigma_s, float df, Buf& buf)
    {
        ensureSizeIsEnough(theta.size(), buf.theta_masked);
        buf.theta_masked.setTo(Scalar::all(0));
        theta.copyTo(buf.theta_masked, cluster);

        times(src_joint_f, buf.theta_masked, buf.mul);

        const Size nsz = Size(saturate_cast<int>(buf.mul.cols * (1.0 / df)), saturate_cast<int>(buf.mul.rows * (1.0 / df)));

        ensureSizeIsEnough(nsz, buf.numerator);
        resize(buf.mul, buf.numerator, Size(), 1.0 / df, 1.0 / df);

        ensureSizeIsEnough(nsz, buf.denominator);
        resize(buf.theta_masked, buf.denominator, Size(), 1.0 / df, 1.0 / df);

        h_filter(buf.numerator, buf.numerator_filtered, sigma_s / df);
        h_filter(buf.denominator, buf.denominator_filtered, sigma_s / df);

        rdivide(buf.numerator_filtered, buf.denominator_filtered, dst);
    }

	void calcEta_single_channel(const Mat_<float>& src_joint_f, const Mat_<float>& theta, const Mat_<uchar>& cluster, Mat_<float>& dst, float sigma_s, float df, Buf& buf)
	{
		ensureSizeIsEnough(theta.size(), buf.theta_masked);
		buf.theta_masked.setTo(Scalar::all(0));
		theta.copyTo(buf.theta_masked, cluster);

		times(src_joint_f, buf.theta_masked, buf.mul_single_channel);

		const Size nsz = Size(saturate_cast<int>(buf.mul_single_channel.cols * (1.0 / df)), saturate_cast<int>(buf.mul_single_channel.rows * (1.0 / df)));

		ensureSizeIsEnough(nsz, buf.numerator);
		resize(buf.mul_single_channel, buf.numerator_single_channel, Size(), 1.0 / df, 1.0 / df);

		ensureSizeIsEnough(nsz, buf.denominator);
		resize(buf.theta_masked, buf.denominator, Size(), 1.0 / df, 1.0 / df);

		h_filter(buf.numerator_single_channel, buf.numerator_filtered_single_channel, sigma_s / df);
		h_filter(buf.denominator, buf.denominator_filtered, sigma_s / df);

		rdivide(buf.numerator_filtered_single_channel, buf.denominator_filtered, dst);
	}

    void AdaptiveManifoldFilterImpl::buildManifoldsAndPerformFiltering(const Mat_<Point3f>& eta_k, const Mat_<uchar>& cluster_k, int current_tree_level)
    {
        // Compute downsampling factor

        double df = min(sigma_s_ / 4.0, 256.0 * sigma_r_);
        df = floor_to_power_of_two(df);
        df = max(1.0, df);

        // Splatting: project the pixel values onto the current manifold eta_k

        if (eta_k.rows == src_joint_f_.rows)
        {
            ensureSizeIsEnough(src_joint_f_.size(), buf_.X);
            subtract(src_joint_f_, eta_k, buf_.X);

            const Size nsz = Size(saturate_cast<int>(eta_k.cols * (1.0 / df)), saturate_cast<int>(eta_k.rows * (1.0 / df)));
            ensureSizeIsEnough(nsz, buf_.eta_k_small);
            resize(eta_k, buf_.eta_k_small, Size(), 1.0 / df, 1.0 / df);
        }
        else
        {
            ensureSizeIsEnough(eta_k.size(), buf_.eta_k_small);
            eta_k.copyTo(buf_.eta_k_small);

            ensureSizeIsEnough(src_joint_f_.size(), buf_.eta_k_big);
            resize(eta_k, buf_.eta_k_big, src_joint_f_.size());

            ensureSizeIsEnough(src_joint_f_.size(), buf_.X);
            subtract(src_joint_f_, buf_.eta_k_big, buf_.X);
        }

        // Project pixel colors onto the manifold -- Eq. (3), Eq. (5)

        ensureSizeIsEnough(buf_.X.size(), buf_.X_squared);
        multiply(buf_.X, buf_.X, buf_.X_squared);

        channelsSum(buf_.X_squared, buf_.pixel_dist_to_manifold_squared);

        phi(buf_.pixel_dist_to_manifold_squared, buf_.gaussian_distance_weights, sigma_r_over_sqrt_2_);

        times(src_f_, buf_.gaussian_distance_weights, buf_.Psi_splat);

        const Mat_<float>& Psi_splat_0 = buf_.gaussian_distance_weights;

        // Save min distance to later perform adjustment of outliers -- Eq. (10)

        min(min_pixel_dist_to_manifold_squared_, buf_.pixel_dist_to_manifold_squared, min_pixel_dist_to_manifold_squared_);

        // Blurring: perform filtering over the current manifold eta_k

        catCn(buf_.Psi_splat, Psi_splat_0, buf_.Psi_splat_joined);

        ensureSizeIsEnough(buf_.eta_k_small.size(), buf_.Psi_splat_joined_resized);
        resize(buf_.Psi_splat_joined, buf_.Psi_splat_joined_resized, buf_.eta_k_small.size());

        RF_filter(buf_.Psi_splat_joined_resized, buf_.eta_k_small, buf_.blurred_projected_values, static_cast<float>(sigma_s_ / df), sigma_r_over_sqrt_2_, buf_);

        split_3_1(buf_.blurred_projected_values, buf_.w_ki_Psi_blur, buf_.w_ki_Psi_blur_0);

        // Slicing: gather blurred values from the manifold

        // Since we perform splatting and slicing at the same points over the manifolds,
        // the interpolation weights are equal to the gaussian weights used for splatting.

        const Mat_<float>& w_ki = buf_.gaussian_distance_weights;

        ensureSizeIsEnough(src_f_.size(), buf_.w_ki_Psi_blur_resized);
        resize(buf_.w_ki_Psi_blur, buf_.w_ki_Psi_blur_resized, src_f_.size());
        times(buf_.w_ki_Psi_blur_resized, w_ki, buf_.w_ki_Psi_blur_resized);
        add(sum_w_ki_Psi_blur_, buf_.w_ki_Psi_blur_resized, sum_w_ki_Psi_blur_);

        ensureSizeIsEnough(src_f_.size(), buf_.w_ki_Psi_blur_0_resized);
        resize(buf_.w_ki_Psi_blur_0, buf_.w_ki_Psi_blur_0_resized, src_f_.size());
        times(buf_.w_ki_Psi_blur_0_resized, w_ki, buf_.w_ki_Psi_blur_0_resized);
        add(sum_w_ki_Psi_blur_0_, buf_.w_ki_Psi_blur_0_resized, sum_w_ki_Psi_blur_0_);

        // Compute two new manifolds eta_minus and eta_plus

        if (current_tree_level < cur_tree_height_)
        {
            // Algorithm 1, Step 2: compute the eigenvector v1
            const Mat_<float> nX(src_joint_f_.size().area(), 3, (float*) buf_.X.data);

            ensureSizeIsEnough(1, nX.cols, buf_.rand_vec);
            rng_.fill(buf_.rand_vec, RNG::UNIFORM, -0.5, 0.5);

            computeEigenVector(nX, cluster_k, buf_.v1, num_pca_iterations_, buf_.rand_vec, buf_);

            // Algorithm 1, Step 3: Segment pixels into two clusters -- Eq. (6)

            ensureSizeIsEnough(nX.rows, buf_.v1.rows, buf_.Nx_v1_mult);
            gemm(nX, buf_.v1, 1.0, noArray(), 0.0, buf_.Nx_v1_mult, GEMM_2_T);

            const Mat_<float> dot(src_joint_f_.rows, src_joint_f_.cols, (float*) buf_.Nx_v1_mult.data);

            Mat_<uchar>& cluster_minus = buf_.cluster_minus[current_tree_level];
            ensureSizeIsEnough(dot.size(), cluster_minus);
            compare(dot, 0, cluster_minus, CMP_LT);
            bitwise_and(cluster_minus, cluster_k, cluster_minus);

            Mat_<uchar>& cluster_plus = buf_.cluster_plus[current_tree_level];
            ensureSizeIsEnough(dot.size(), cluster_plus);
            compare(dot, 0, cluster_plus, CMP_GT);
            bitwise_and(cluster_plus, cluster_k, cluster_plus);

            // Algorithm 1, Step 4: Compute new manifolds by weighted low-pass filtering -- Eq. (7-8)

            ensureSizeIsEnough(w_ki.size(), buf_.theta);
            buf_.theta.setTo(Scalar::all(1.0));
            subtract(buf_.theta, w_ki, buf_.theta);

            Mat_<Point3f>& eta_minus = buf_.eta_minus[current_tree_level];
            calcEta(src_joint_f_, buf_.theta, cluster_minus, eta_minus, sigma_s_, df, buf_);

            Mat_<Point3f>& eta_plus = buf_.eta_plus[current_tree_level];
            calcEta(src_joint_f_, buf_.theta, cluster_plus, eta_plus, sigma_s_, df, buf_);

            // Algorithm 1, Step 5: recursively build more manifolds.

            buildManifoldsAndPerformFiltering(eta_minus, cluster_minus, current_tree_level + 1);
            buildManifoldsAndPerformFiltering(eta_plus, cluster_plus, current_tree_level + 1);
        }
    }

	void AdaptiveManifoldFilterImpl::buildManifoldsAndPerformFiltering_single_channel(const Mat_<float>& eta_k, const Mat_<uchar>& cluster_k, int current_tree_level)
	{
		// Compute downsampling factor
        double df = min(sigma_s_ / 4.0, 256.0 * sigma_r_);
		df = floor_to_power_of_two(df);
		df = max(1.0, df);

		// Splatting: project the pixel values onto the current manifold eta_k
		if (eta_k.rows == src_joint_f_single_channel.rows)
		{
			ensureSizeIsEnough(src_joint_f_single_channel.size(), buf_.X_single_channel);
			subtract(src_joint_f_single_channel, eta_k, buf_.X_single_channel);

			const Size nsz = Size(saturate_cast<int>(eta_k.cols * (1.0 / df)), saturate_cast<int>(eta_k.rows * (1.0 / df)));
			ensureSizeIsEnough(nsz, buf_.eta_k_small_single_channel);
			resize(eta_k, buf_.eta_k_small_single_channel, Size(), 1.0 / df, 1.0 / df);
		}
		else
		{
			ensureSizeIsEnough(eta_k.size(), buf_.eta_k_small_single_channel);
			eta_k.copyTo(buf_.eta_k_small_single_channel);

			ensureSizeIsEnough(src_joint_f_single_channel.size(), buf_.eta_k_big_single_channel);
			resize(eta_k, buf_.eta_k_big_single_channel, src_joint_f_single_channel.size());

			ensureSizeIsEnough(src_joint_f_single_channel.size(), buf_.X_single_channel);
			subtract(src_joint_f_single_channel, buf_.eta_k_big_single_channel, buf_.X_single_channel);
		}

		// Project pixel colors onto the manifold -- Eq. (3), Eq. (5)
		ensureSizeIsEnough(buf_.X_single_channel.size(), buf_.X_squared_single_channel);
		multiply(buf_.X_single_channel, buf_.X_single_channel, buf_.X_squared_single_channel);

		buf_.X_squared_single_channel.copyTo (buf_.pixel_dist_to_manifold_squared);

		phi(buf_.pixel_dist_to_manifold_squared, buf_.gaussian_distance_weights, sigma_r_over_sqrt_2_);

		times(src_f_single_channel, buf_.gaussian_distance_weights, buf_.Psi_splat_single_channel);

		const Mat_<float>& Psi_splat_0 = buf_.gaussian_distance_weights;

		// Save min distance to later perform adjustment of outliers -- Eq. (10)
		min(min_pixel_dist_to_manifold_squared_, buf_.pixel_dist_to_manifold_squared, min_pixel_dist_to_manifold_squared_);

		// Blurring: perform filtering over the current manifold eta_k

		catCn2Channels(buf_.Psi_splat_single_channel, Psi_splat_0, buf_.Psi_splat_joined_single_channel);

		ensureSizeIsEnough(buf_.eta_k_small_single_channel.size(), buf_.Psi_splat_joined_resized_single_channel);
		resize(buf_.Psi_splat_joined_single_channel, buf_.Psi_splat_joined_resized_single_channel, buf_.eta_k_small_single_channel.size());

		RF_filter_single_channel(buf_.Psi_splat_joined_resized_single_channel, buf_.eta_k_small_single_channel, buf_.blurred_projected_values_single_channel, static_cast<float>(sigma_s_ / df), sigma_r_over_sqrt_2_, buf_);

		split_1_1(buf_.blurred_projected_values_single_channel, buf_.w_ki_Psi_blur_single_channel, buf_.w_ki_Psi_blur_0);

		// Slicing: gather blurred values from the manifold

		// Since we perform splatting and slicing at the same points over the manifolds,
		// the interpolation weights are equal to the gaussian weights used for splatting.
		const Mat_<float>& w_ki = buf_.gaussian_distance_weights;

		ensureSizeIsEnough(src_f_single_channel.size(), buf_.w_ki_Psi_blur_resized_single_channel);
		resize(buf_.w_ki_Psi_blur_single_channel, buf_.w_ki_Psi_blur_resized_single_channel, src_f_single_channel.size());
		times(buf_.w_ki_Psi_blur_resized_single_channel, w_ki, buf_.w_ki_Psi_blur_resized_single_channel);
		add(sum_w_ki_Psi_blur_single_channel, buf_.w_ki_Psi_blur_resized_single_channel, sum_w_ki_Psi_blur_single_channel);

		ensureSizeIsEnough(src_f_single_channel.size(), buf_.w_ki_Psi_blur_0_resized);
		resize(buf_.w_ki_Psi_blur_0, buf_.w_ki_Psi_blur_0_resized, src_f_single_channel.size());
		times(buf_.w_ki_Psi_blur_0_resized, w_ki, buf_.w_ki_Psi_blur_0_resized);
		add(sum_w_ki_Psi_blur_0_, buf_.w_ki_Psi_blur_0_resized, sum_w_ki_Psi_blur_0_);

		// Compute two new manifolds eta_minus and eta_plus

		if (current_tree_level < cur_tree_height_)
		{
			// Algorithm 1, Step 2: compute the eigenvector v1
			/*const Mat_<float> nX(src_joint_f_single_channel.size().area(), 3, (float*) buf_.X_single_channel.data);*/
			const Mat_<float> nX(src_joint_f_single_channel.size().area(), 1, (float*) buf_.X_single_channel.data);
			ensureSizeIsEnough(1, nX.cols, buf_.rand_vec);
			rng_.fill(buf_.rand_vec, RNG::UNIFORM, -0.5, 0.5);

			computeEigenVector(nX, cluster_k, buf_.v1, num_pca_iterations_, buf_.rand_vec, buf_);

			// Algorithm 1, Step 3: Segment pixels into two clusters -- Eq. (6)

			ensureSizeIsEnough(nX.rows, buf_.v1.rows, buf_.Nx_v1_mult);
			gemm(nX, buf_.v1, 1.0, noArray(), 0.0, buf_.Nx_v1_mult, GEMM_2_T);

			const Mat_<float> dot(src_joint_f_single_channel.rows, src_joint_f_single_channel.cols, (float*) buf_.Nx_v1_mult.data);

			Mat_<uchar>& cluster_minus = buf_.cluster_minus[current_tree_level];
			ensureSizeIsEnough(dot.size(), cluster_minus);
			compare(dot, 0, cluster_minus, CMP_LT);
			bitwise_and(cluster_minus, cluster_k, cluster_minus);

			Mat_<uchar>& cluster_plus = buf_.cluster_plus[current_tree_level];
			ensureSizeIsEnough(dot.size(), cluster_plus);
			compare(dot, 0, cluster_plus, CMP_GT);
			bitwise_and(cluster_plus, cluster_k, cluster_plus);

			// Algorithm 1, Step 4: Compute new manifolds by weighted low-pass filtering -- Eq. (7-8)

			ensureSizeIsEnough(w_ki.size(), buf_.theta);
			buf_.theta.setTo(Scalar::all(1.0));
			subtract(buf_.theta, w_ki, buf_.theta);

			Mat_<float>& eta_minus = buf_.eta_minus_single_channel[current_tree_level];
			calcEta_single_channel(src_joint_f_single_channel, buf_.theta, cluster_minus, eta_minus, sigma_s_, df, buf_);

			Mat_<float>& eta_plus = buf_.eta_plus_single_channel[current_tree_level];
			calcEta_single_channel(src_joint_f_single_channel, buf_.theta, cluster_plus, eta_plus, sigma_s_, df, buf_);

			// Algorithm 1, Step 5: recursively build more manifolds.

			buildManifoldsAndPerformFiltering_single_channel(eta_minus, cluster_minus, current_tree_level + 1);
			buildManifoldsAndPerformFiltering_single_channel(eta_plus, cluster_plus, current_tree_level + 1);
		}
	}
}

Ptr<AdaptiveManifoldFilter> cv::AdaptiveManifoldFilter::create()
{
    return new AdaptiveManifoldFilterImpl;
}
