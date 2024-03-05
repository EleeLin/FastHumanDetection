/* --- --- ---
 * Copyright (C) 2008--2010 Idiap Research Institute (.....@idiap.ch)
 * All rights reserved.
 * 
 * Redistribution and use in source and binary forms, with or without
 * modification, are permitted provided that the following conditions
 * are met:
 * 1. Redistributions of source code must retain the above copyright
 *    notice, this list of conditions and the following disclaimer.
 * 2. Redistributions in binary form must reproduce the above copyright
 *    notice, this list of conditions and the following disclaimer in the
 *    documentation and/or other materials provided with the distribution.
 * 3. The name of the author may not be used to endorse or promote products
 *    derived from this software without specific prior written permission.
 * 
 * THIS SOFTWARE IS PROVIDED BY THE AUTHOR ``AS IS'' AND ANY EXPRESS OR
 * IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED WARRANTIES
 * OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE DISCLAIMED.
 * IN NO EVENT SHALL THE AUTHOR BE LIABLE FOR ANY DIRECT, INDIRECT,
 * INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT
 * NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE,
 * DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY
 * THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
 * (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF
 * THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
*/
// bgsub_detect.cpp : Defines the entry point for the console application.
//


//#include "BackgroundSubtractionAPI.h"		/* API header file of background subtraction algorithm */
//#include "BackgroundSubtraction.h"		/* header file of background subtraction algorithm */
#include "MultiLayerBGS.h"		/* header file of background subtraction algorithm */

#include "CmdLine.h"
#include "Timer.h"
#include "FileList.h"
#include "Camera.h"

#include "cv.h"
#include "highgui.h"
#include "cxcore.h"

#ifndef CHECK_STRING
#define CHECK_STRING(x)  {if(x && strlen(x)==0) {x=NULL;} }
#endif

#include "utilities.h"

#if !(defined(BGSUB_LEARN) || defined(BGSUB_DETECT))
#error "BGSUB_LEARN or BGSUB_DETECT should be defined."
#endif
#if (defined(BGSUB_LEARN) && defined(BGSUB_DETECT))
#error "Only one of BGSUB_LEARN or BGSUB_DETECT should be defined."
#endif

int main(int argc, char* argv[])
{
	Torch::CmdLine cmd;

	cmd.addText("\n******************************************************************************");
#ifdef BGSUB_LEARN
	cmd.addText("**   PROGRAM: Learning Background Model from a Set of Images or a Video     **");
#else
	cmd.addText("**   PROGRAM: Foreground Detection from a Set of Images or a Video          **");
#endif
	cmd.addText("**                                                                          **");
	cmd.addText("**             Author:   Dr. Jian Yao                                       **");
	cmd.addText("**                       IDIAP Research Institute                           **");
	cmd.addText("**                       CH-1920, Martigny, Switzerland                     **");
	cmd.addText("**             Emails:   Jian.Yao@idiap.ch                                  **");
	cmd.addText("**                       jianyao75@gmail.com                                **");
	cmd.addText("**             Date:     October 29, 2008                                   **");
	cmd.addText("******************************************************************************");

	char msg[2048];

    cmd.addText("\nArguments:");
    char* image_list_or_video_fn;
    char* bg_model_fn;

    cmd.addSCmdArg("images list or video", &image_list_or_video_fn,
                   "the list of images or an video to be used for learning background model");
    cmd.addSCmdArg("background model", &bg_model_fn,
                   "the output file name of learned background model");

    cmd.addText("\nOptions:");

#ifdef BGSUB_LEARN
    char indent_space[] = "                         ";
    cmd.addText("\n *Saved model type:");
    int bg_model_save_type = 2;
    cmd.addICmdOption("-bgmt", &bg_model_save_type, 2,
                      "the saving type of the learned background model");
    sprintf(msg, "%s   0 - background model information (pixel by pixel)", indent_space);
    cmd.addText(msg);
    sprintf(msg, "%s   1 - background model parameters", indent_space);
    cmd.addText(msg);
    sprintf(msg, "%s   2 - both background information (pixel by pixel) and parameters", indent_space);
    cmd.addText(msg);
    char* bg_model_preload = NULL;
    cmd.addSCmdOption("-incr", &bg_model_preload, "",
                      "An existing background model to preload (if you need to learn on multiple sequences)");
#endif

    cmd.addText("\n *Frames selection:");
    int start_frame_idx = 0;
    int end_frame_idx = 1000000;
    int skipped_frames_num = 0;
    cmd.addICmdOption("-sf", &start_frame_idx, 0,
                      "the start frame index for foreground detection");
    cmd.addICmdOption("-ef", &end_frame_idx, 1000000,
                      "the end frame index for foreground detection");
    cmd.addICmdOption("-sfn", &skipped_frames_num, 0,
                      "the number of skipped frames between two used frames for detection");

    cmd.addText("\n *Pre-processing:");
    real width_scale = 1.0;
    real height_scale = 1.0;
    real noise_sigma = 0;
    cmd.addRCmdOption("-xs", &width_scale, 1.0,
                      "the image scaling factor at the x-direction (width)");
    cmd.addRCmdOption("-ys", &height_scale, 1.0,
                      "the image scaling factor at the y-direction (height)");
    cmd.addRCmdOption("-ngaus", &noise_sigma, 0,
                      "the Gaussian sigma to remove noise on original images");

    cmd.addText("\n *Learning rates:");
#ifdef BGSUB_DETECT
    bool disable_learning = false;
    cmd.addBCmdOption("-nolearn", &disable_learning, disable_learning,
                      "fully disable online learning of background [false]");
#endif
    real mode_learn_rate_per_second = 0.01;
    real weight_learn_rate_per_second = 0.01;
    real init_mode_weight = 0.001;
    real weight_updating_constant = 5.0;
    real frame_duration = 1.0 / 25.0;
	real lbp_level_weight_exponential_decaying = 1.0;

#ifdef BGSUB_LEARN
    mode_learn_rate_per_second = 0.5;
    weight_learn_rate_per_second = 0.5;
    init_mode_weight = 0.05;
#endif
	
	cmd.addRCmdOption("-lbplw", &lbp_level_weight_exponential_decaying, 1.0,
		"the exponential decaying constant of the multi-level lbp");
    cmd.addRCmdOption("-mlr", &mode_learn_rate_per_second, mode_learn_rate_per_second,
                      "the learning rate per second for the background modes in the detection processing");
    cmd.addRCmdOption("-wlr", &weight_learn_rate_per_second, weight_learn_rate_per_second,
                      "the learning rate per second for the background mode weights in the detection processing");
    cmd.addRCmdOption("-initw", &init_mode_weight, init_mode_weight,
                      "the initial mode weight for a newly added background mode in the detection processing");
    cmd.addRCmdOption("-wuc", &weight_updating_constant, weight_updating_constant,
                      "the mode weight hysteresis (increase/decrease) updating constant");
    cmd.addRCmdOption("-fdur", &frame_duration, frame_duration,
                      "the duration between two used frames");

    cmd.addText("\n *Model parameters:");
    real texture_weight = 0.5;
    real bg_mode_percent = 0.6;
    real bg_prob_updating_threshold = 0.2;
	real bg_prob_threshold = 0.2;
	real bg_prob_threshold_upperbound = 0.6;
	real bg_prob_threshold_lowerbound = 0.1;
	int frame_accelerating_dist_threshold = 5;
    int max_mode_num = 5;
    real shadow_rate = 0.6;
    real highlight_rate = 1.2;
    int robust_LBP_constant = 3;
    real min_noised_angle = 10.0 / 180.0 * PI;
	bool LTP_used = false;
	bool GRD_used = false;
	bool FRM_used = false;
	int frame_skipped_interval = 1;
	int frame_skipped_first = 0;
	
    cmd.addICmdOption("-maxm", &max_mode_num, 5,
                      "maximal mode number for each pixel");
    cmd.addRCmdOption("-tw", &texture_weight, 0.5,
                      "distance weight by using LBP texture feature");
    cmd.addRCmdOption("-bmp", &bg_mode_percent, 0.6,
                      "reliable background mode percent for foreground detection");
    cmd.addRCmdOption("-bgut", &bg_prob_updating_threshold, 0.2,
                      "background/foreground distance threshold in the modeling process");
    cmd.addRCmdOption("-sr", &shadow_rate, 0.6,
                      "robust shadow rate for color illumination changes");
    cmd.addRCmdOption("-hr", &highlight_rate, 1.2,
                      "robust highlight rate for color illumination changes");
    cmd.addBCmdOption("-ltp", &LTP_used, false,
                      "use the LTP for texture description [false]");
	cmd.addBCmdOption("-grd", &GRD_used,false,
					  "use the gradient for accelerating the computation [false]");
	cmd.addBCmdOption("-frm", &FRM_used,false,
					"use the frame skipping for accelerating the computation [false]");
	cmd.addICmdOption("-fsi",&frame_skipped_interval,1,
					"the parameter deciding interval used for skipping");
	cmd.addICmdOption("-fsf",&frame_skipped_first,0,
					"the parameter deciding when to start using the frame skipping");
	cmd.addICmdOption("-rc", &robust_LBP_constant, 3,
                      "robust color offset in each channel for LBP or LTP");
    cmd.addRCmdOption("-mna", &min_noised_angle, min_noised_angle,
                      "minimal noised angle between the mode color and the observed color");
	cmd.addRCmdOption("-bgt", &bg_prob_threshold, 0.2,
		              "background/foreground distance threshold in the detection process");
	cmd.addRCmdOption("-bgtu",&bg_prob_threshold_upperbound,0.6,
		              "background/foreground distance upperbound of threshold in the detection process");
	cmd.addRCmdOption("-bgtl",&bg_prob_threshold_lowerbound,0.1,
					  "background/foreground distance lowerbound of threshold in the detection process");
	cmd.addICmdOption("-fadt",&frame_accelerating_dist_threshold,5,
					  "the threshold to judge whether or not to compute the dist in the current frame");

	cmd.addText("\n *Post-processing:");

    int pattern_neig_half_size = 4;
    real pattern_neig_gaus_sigma = 3.0;
    real bilater_filter_sigma_s = 3.0;
    real bilater_filter_sigma_r = 0.1;
	bool AMF_used = false;
	real AMFSigmaS = 16.0;
	real AMFSigmaR = 0.2;
	int AMFTreeHeight = -1;
	int AMFNumPcaIterations = 1;
	
    cmd.addICmdOption("-phs", &pattern_neig_half_size, 5,
                      "neighboring half size of pattern window for gaussian filtering on distance map to remove noise");
    cmd.addRCmdOption("-pgs", &pattern_neig_gaus_sigma, 3.0,
                      "gaussian sigma used to remove noise applied on distance map");
    cmd.addRCmdOption("-bfss", &bilater_filter_sigma_s, 3.0,
                      "spatial sigma for cross bilateral filter used to remove noise on distance map");
    cmd.addRCmdOption("-bfsr", &bilater_filter_sigma_r, 0.1,
                      "normalized color radius sigma for cross bilateral filter used to remove noise on distance map");
	
	cmd.addBCmdOption("-amf", &AMF_used, false,
                      "use the AMF to denoise the forg_prob_image [false]");
    cmd.addRCmdOption("-amfss", &AMFSigmaS, 16.0,
                      "spatial standard deviation for adaptive manifold filter used to remove noise on distance map");
	cmd.addRCmdOption("-amfsr", &AMFSigmaR, 0.2,
                      "range standard deviation for adaptive manifold filter used to remove noise on distance map");
	cmd.addICmdOption("-amfth", &AMFTreeHeight, -1,
                      "Height of the manifold tree [-1 (automatically computed)]");
	cmd.addICmdOption("-amfnpi", &AMFNumPcaIterations, 1,
                      "number of iterations to computed the eigenvector v1 in the adaptive manifold filter");
  
    cmd.addText("\n Warping with vertical vanishing point:");
    char* cam_paras_fn = NULL;
    char* updated_cam_paras_fn = NULL;
    cmd.addSCmdOption("-cam", &cam_paras_fn, "",
                      "the camera parameters file (3x4 projection matrix + distortion coefficients) (YAML format)");
    cmd.addSCmdOption("-ucam", &updated_cam_paras_fn, "",
                      "the updated camera parameters file (YAML format)");

    cmd.addText("\n Outputs:");

    char* output_warped_mask_fn = NULL;
    cmd.addSCmdOption("-owm", &output_warped_mask_fn, "",
                      "the output warped mask file");
#ifdef BGSUB_DETECT
    char* output_dir = NULL;
    bool export_org_img = false;
    bool export_fg_img = false;
    bool export_fg_mask_img = false;
    bool export_bg_img = false;
    bool export_fg_prob_img = false;
    bool export_merged_img = false;
	bool export_gradient_img = false;
	char* output_video_filename = NULL;
	cmd.addText("");
	cmd.addSCmdOption("-ovideo", &output_video_filename, "",
		"the detection output video file name");
	cmd.addText("");
    cmd.addSCmdOption("-od", &output_dir, "",
                      "the detection output directory");
    cmd.addBCmdOption("-oog", &export_org_img, false,
                      "export the original image [false]");
    cmd.addBCmdOption("-ofi", &export_fg_img, false,
                      "export the foreground image [false]");
    cmd.addBCmdOption("-ofmi", &export_fg_mask_img, false,
                      "export the foreground mask image [false]");
    cmd.addBCmdOption("-ofpi", &export_fg_prob_img, false,
                      "export the foreground probability image [false]");
    cmd.addBCmdOption("-obi", &export_bg_img, false,
                      "export the background image [false]");
	cmd.addBCmdOption("-egi", &export_gradient_img, false,
					  "export the gradient image [false]");
    cmd.addBCmdOption("-omerged", &export_merged_img, false,
                      "export the merged image [false]");

#endif

    cmd.addText("\n *Others:");
    bool display_results = false;
    cmd.addBCmdOption("-os", &display_results, false,
                      "displaying the learning results");

    cmd.addText("");

    /* Read the command line */
    cmd.read(argc, argv);

	//image_list_or_video_fn = "G:\\SERVERS\\FTP\\CVRS\\Projects\\VideoSurveillance\\Videos\\TestVideo1.avi";
	//bg_model_fn = "testvideo1.bgm";
	//display_results = true;
	//output_video_filename = "TestVideo1-results.avi";

    CHECK_STRING(cam_paras_fn);
    CHECK_STRING(updated_cam_paras_fn);
    CHECK_STRING(output_warped_mask_fn);
#ifdef BGSUB_LEARN
	CHECK_STRING(bg_model_preload);
#endif
#ifdef BGSUB_DETECT
    CHECK_STRING(output_dir);
	CHECK_STRING(output_video_filename);
#endif

    const char* disp_win_name = "LT: Original | RT: Background | LB: Distance | RB: Foreground";

    /* check the input data: image list or avi video */
    CvCapture* VIDEO = NULL; /* the opencv video capture handle */
    VIDEO = cvCaptureFromFile(image_list_or_video_fn);
    CFileList* LIST = NULL;
    if (VIDEO == NULL) {
        LIST = new CFileList(image_list_or_video_fn);
    } else {
        skip_first_frames(VIDEO, start_frame_idx);
    }

    CTimer Timer;

    /* load the images */
    IplImage* org_img = VIDEO ? cvQueryFrame(VIDEO) : cvLoadImage(LIST->GetFileName(0));

    Camera* CAM = NULL;
    CvSize img_size;
    if (cam_paras_fn) {
        CAM = new Camera(org_img);
        CAM->loadCalibration(cam_paras_fn);
        CAM->computeTransforms(width_scale, height_scale);
        img_size = cvGetSize(CAM->undistorted_image);
        if (updated_cam_paras_fn)
            CAM->SaveUpdatedCalibration(updated_cam_paras_fn);

        if (output_warped_mask_fn)
            cvSaveImage(output_warped_mask_fn, CAM->mask);
    } else
        img_size = cvSize(cvCeil((double) org_img->width * width_scale),
                          cvCeil((double) org_img->height * height_scale));

    /* allocate memories */
    IplImage* img = cvCreateImage(img_size, org_img->depth, org_img->nChannels);//the original image	
	IplImage* old_img = cvCreateImage(img_size, org_img->depth, org_img->nChannels);//the index (n-2) image

	IplImage* fg_img = cvCreateImage(img_size, org_img->depth, org_img->nChannels);//the foreground
/*	IplImage* fg_img_old = cvCreateImage(img_size, org_img->depth, org_img->nChannels);//the foreground*/
    
	IplImage* fg_prob_img3 = cvCreateImage(img_size, org_img->depth, org_img->nChannels);
	IplImage* fg_prob_img3_old = cvCreateImage(img_size, org_img->depth, org_img->nChannels);    // get dist_old
    
	IplImage* fg_prob_img = cvCreateImage(img_size, org_img->depth, 1);
	IplImage* fg_mask_old_img_output = cvCreateImage(img_size, org_img->depth, 1);    // get mask_old(n-2) to aovid the mistakes of pointers
    
	IplImage* bg_img = cvCreateImage(img_size, org_img->depth, org_img->nChannels);
/*    IplImage* bg_img_old = cvCreateImage(img_size, org_img->depth, org_img->nChannels);*/

	IplImage* fg_mask_img = cvCreateImage(img_size, org_img->depth, 1);
	IplImage* fg_mask_img_output = cvCreateImage(img_size, org_img->depth, 1);    // get mask to avoid the mistakes of pointers
	
	IplImage* cur_img_gradient = cvCreateImage(img_size, org_img->depth, 1);
	
	IplImage** ppSkippedImg = new IplImage* [frame_skipped_interval - 1];//1
	IplImage** ppSkippedBg = new IplImage* [frame_skipped_interval - 1];
	IplImage** ppSkippedDist = new IplImage* [frame_skipped_interval - 1];
	IplImage** ppSkippedFg = new IplImage* [frame_skipped_interval - 1];
	IplImage** ppMergedImg = new IplImage* [frame_skipped_interval - 1];
    
	for (int i = 0; i < frame_skipped_interval - 1; i++)
	{
		ppSkippedImg[i] = cvCreateImage(img_size,  org_img->depth, org_img->nChannels);
		ppSkippedBg[i] = cvCreateImage(img_size,  org_img->depth, org_img->nChannels);
		ppSkippedDist[i] = cvCreateImage(img_size,  org_img->depth, org_img->nChannels);
		ppSkippedFg[i] = cvCreateImage(img_size,  org_img->depth, org_img->nChannels);
		ppMergedImg[i] = cvCreateImage(cvSize(img_size.width * 2, img_size.height * 2),  org_img->depth, org_img->nChannels);
	}

	CvVideoWriter *writer = NULL;
	IplImage* merged_img = NULL;

#ifdef BGSUB_DETECT
    if (display_results || (output_dir && export_merged_img))
#else
        if (display_results)
#endif
            merged_img = cvCreateImage(cvSize(img_size.width * 2, img_size.height * 2), org_img->depth, org_img->nChannels);

    if (!VIDEO)
        cvReleaseImage(&org_img);

    /************************************************************************/
    /* INITALIZATION                                                        */
    /************************************************************************/

    CMultiLayerBGS* BGS = new CMultiLayerBGS();

	/* parameter setting */
	BGS->m_nMaxLBPModeNum = max_mode_num;
	BGS->m_fWeightUpdatingConstant = weight_updating_constant;
	BGS->m_fTextureWeight = texture_weight;
	BGS->m_fBackgroundModelPercent = bg_mode_percent;
	BGS->m_nPatternDistSmoothNeigHalfSize = pattern_neig_half_size;
	BGS->m_fPatternDistConvGaussianSigma = pattern_neig_gaus_sigma;
	BGS->m_fPatternColorDistBgThreshold = bg_prob_threshold;
	BGS->m_fPatternColorDistBgThreshold_UpperBound = bg_prob_threshold_upperbound;
	BGS->m_fPatternColorDistBgThreshold_LowerBound = bg_prob_threshold_lowerbound;
	BGS->m_fPatternColorDistBgUpdatedThreshold = bg_prob_updating_threshold;
	BGS->m_fRobustColorOffset = robust_LBP_constant;
	BGS->m_bLTPUsed = LTP_used;
	BGS->m_bGRDUsed = GRD_used;
	BGS->m_bFRMUsed = FRM_used;
	BGS->m_iFrameSkippedFirst = frame_skipped_first; 
	BGS->m_iFrameSkippedInterval = frame_skipped_interval;
	BGS->m_iFrameAcceleratingDistThreshold = frame_accelerating_dist_threshold;
	BGS->m_bAMFUsed = AMF_used;
	BGS->m_fAMFSigmaS = AMFSigmaS;
	BGS->m_fAMFSigmaR = AMFSigmaR;
	BGS->m_nAMFTreeHeight = AMFTreeHeight;
	BGS->m_nAMFNumPcaIterations = AMFNumPcaIterations;
	BGS->m_fMinNoisedAngle = min_noised_angle;
	BGS->m_fRobustShadowRate = shadow_rate;
	BGS->m_fRobustHighlightRate = highlight_rate;
	BGS->m_fSigmaS = bilater_filter_sigma_s;
	BGS->m_fSigmaR = bilater_filter_sigma_r;
	BGS->m_f_lbp_level_weight_exponential_decaying_constant = lbp_level_weight_exponential_decaying;

	/* BGS initialization */
    BGS->Init(img_size.width, img_size.height);

    /* set the foreground mask image pointer */
    BGS->SetForegroundMaskImage(fg_mask_img);

	/* set the gradient image pointer */
	BGS->SetCurrentGradientImage(cur_img_gradient);

    /* set the foreground probability image pointer */
    BGS->SetForegroundProbImage(fg_prob_img);

#ifdef BGSUB_DETECT
    BGS->m_disableLearning = disable_learning;
    /* Load background model: parameters and modeling information */
    BGS->Load(bg_model_fn);
#else
    if (bg_model_preload && strlen(bg_model_preload)>0) 
		BGS->Load(bg_model_preload);
#endif

    /************************************************************************/
    /*  BACKGROUND DETECTION PROCESSING                                     */
    /************************************************************************/

    /* set frame rate using frame duration, for example, 25 frames per second, the frame duration = 1/25 */
    BGS->SetFrameRate(frame_duration);

    /* set main background learning parameters */
    BGS->SetParameters(max_mode_num, mode_learn_rate_per_second, weight_learn_rate_per_second, init_mode_weight);

    // background learning process
    if (display_results)
        cvNamedWindow(disp_win_name);

	double count_time_total = 0.0f;

	if (FRM_used)
	{
		skipped_frames_num = 0;
		// TDOD :
	}

	for (int frame_idx = start_frame_idx; frame_idx < end_frame_idx; frame_idx += skipped_frames_num + frame_skipped_interval) {
	             
/*
        if (!get_next_image(VIDEO, LIST, CAM, img, noise_sigma, frame_idx, skipped_frames_num))
            break; */
     
		if (!frame_idx)
		{
			if (!get_next_image(VIDEO, LIST, CAM, img, noise_sigma, frame_idx, skipped_frames_num))
			{
				break;
			}
		}

		else
		{
			cvCopy (img, old_img);/**/

			/*cvNamedWindow ("debug1");
			cvShowImage ("debug1", old_img);
			cvWaitKey (1);*/

			if (!get_next_image_skipped (VIDEO, LIST, CAM, img, noise_sigma, frame_idx, skipped_frames_num, frame_skipped_interval, ppSkippedImg))
				break;
		}

/*
		cvNamedWindow ("debug");
		cvShowImage ("debug", img);
		cvWaitKey (0);*/

        /* record the start time of background subtraction */
        Timer.Start();

        /* set the new image data */
        BGS->SetRGBInputImage(img);

		/*cvNamedWindow ("debug2");
		cvShowImage ("debug2", img);
		cvWaitKey (1);*/
		
		BGS->SetCurrentFrameNumber((unsigned long)frame_idx);

        /* do background subtraction process */
        BGS->Process ();

// 		if (frame_idx)
// 		{
// 			cvCopy(bg_img,bg_img_old);
// 		}

		BGS->GetBackgroundImage(bg_img);

// 		if (frame_idx)
// 		{
// 			cvCopy(fg_img,fg_img_old);
// 		}		

		BGS->GetForegroundImage(fg_img);

		if (frame_idx)
		{
			cvCopy (fg_mask_img_output, fg_mask_old_img_output);//0
			cvCopy (fg_prob_img3, fg_prob_img3_old);

			/* get foreground mask image */
			BGS->GetForegroundMaskImage(fg_mask_img_output);//2

			/* get foreground probability image */
			BGS->GetForegroundProbabilityImage(fg_prob_img3);

			/*if 0 and 2 belongs to the same property*/

			BGS->IsComputation (fg_mask_old_img_output,fg_mask_img_output);

			for (int i = 0; i < frame_skipped_interval - 1; i++)
			{
				BGS->SetRGBInputImage (ppSkippedImg [i]);
				BGS->SetCurrentFrameNumber((unsigned long)(i + frame_idx - frame_skipped_interval + 1));
				BGS->Process ();
				BGS->GetBackgroundImage(ppSkippedBg[i]);
				BGS->GetForegroundProbabilityImage(ppSkippedDist[i]);
				BGS->GetForegroundImage(ppSkippedFg[i]);

				/*cvNamedWindow ("debug3");
				cvShowImage ("debug3", ppSkippedImg [i]);
				cvWaitKey (1);

				cvNamedWindow ("debug4");
				cvShowImage ("debug4", ppSkippedBg[i]);
				cvWaitKey (1);

				cvNamedWindow ("debug5");
				cvShowImage ("debug5", ppSkippedDist[i]);
				cvWaitKey (1);

				cvNamedWindow ("debug6");
				cvShowImage ("debug6", ppSkippedFg[i]);
				cvWaitKey (1);*/
			}
		}

        /* record the stop time of background subtraction */
        Timer.Stop();
        Timer.PrintElapsedTimeMsg(msg);
		count_time_total+= Timer.GetElapsedSeconds();

        /* print the time */
        printf("Learning - Frame : %d\tTime : %s\n", frame_idx, msg);

        if (display_results
#ifdef BGSUB_DETECT
			|| output_video_filename 
#endif
			) 
		{
            /* get background image */
            //BGS->GetBackgroundImage(bg_img);

            /* get foreground mask image */
          //  BGS->GetForegroundMaskImage(m_pFgMaskImg);

			/*get current gradient image*/
// 			if (BGS->m_bGRDUsed)
// 			{			
// 				BGS->GetCurrentGradientImage(cur_img_gradient);
// 			}

            /* get foreground image */
           // BGS->GetForegroundImage(fg_img);

            /* get foreground probability image */
           // BGS->GetForegroundProbabilityImage(fg_prob_img3);

			/*get the current gradient image*/
            /* merge the above 4 images into one image for display */

			
		/*cvNamedWindow ("debug7");
		cvShowImage ("debug7", img);
		cvWaitKey (1);
			
		cvNamedWindow ("debug8");
		cvShowImage ("debug8", bg_img);
		cvWaitKey (1);
			
		cvNamedWindow ("debug9");
		cvShowImage ("debug9", fg_prob_img3);
		cvWaitKey (1);
			
		cvNamedWindow ("debug10");
		cvShowImage ("debug10", fg_img);
		cvWaitKey (1);*/

			if (frame_idx)
			{
				for (int i = 0; i < frame_skipped_interval-1; i++)
				{
					BGS->MergeImages (4, ppSkippedImg[i], ppSkippedBg[i], ppSkippedDist[i], ppSkippedFg[i], ppMergedImg[i]);
				}

				BGS->MergeImages(4, img, bg_img, fg_prob_img3, fg_img, merged_img);
			}

			if (frame_idx == 0)
                BGS->MergeImages(4, img, bg_img, fg_prob_img3, fg_img, merged_img);

#ifdef BGSUB_DETECT
			if (output_video_filename) 
			{
				if ( !writer ) {
					writer=cvCreateVideoWriter(output_video_filename,CV_FOURCC('M','J','P','G'),30,cvSize(merged_img->width, merged_img->height));
					if ( !writer ) {
						fprintf(stderr,"Can't create the video writer");
						return 0;
					}
				}
				cvWriteFrame(writer,merged_img);
			}
#endif

            /* show merged image */
			if (!frame_idx)
			{
				cvShowImage(disp_win_name, merged_img);

				cvWaitKey(10);
			}

			else
			{
				for (int i = 0;i<frame_skipped_interval - 1;i++)
				{
					cvShowImage(disp_win_name, ppMergedImg[i]);
					cvWaitKey(10);
				}

				cvShowImage(disp_win_name, merged_img);
				cvWaitKey(10);
			}
        }

#ifdef BGSUB_DETECT
        if (output_dir) {
            if (export_org_img) {
                char* file_name = get_file_name(output_dir, "org_img", "jpg", frame_idx);
                cvSaveImage(file_name, img);
                delete [] file_name;
            }
            if (export_fg_img) {
                if (!display_results)
                    BGS->GetForegroundImage(fg_img);
                char* file_name = get_file_name(output_dir, "fg_img", "jpg", frame_idx);
                cvSaveImage(file_name, fg_img);
                delete [] file_name;
            }
            if (export_fg_mask_img) {
                if (!display_results)
                    BGS->GetForegroundMaskImage(fg_mask_img);
                char* file_name = get_file_name(output_dir, "fg_mask_img", "jpg", frame_idx);
                cvSaveImage(file_name, fg_mask_img);
                delete [] file_name;
            }
            if (export_fg_prob_img) {
                char* file_name = get_file_name(output_dir, "fg_prob_img", "jpg", frame_idx);
                cvSaveImage(file_name, fg_prob_img);
                delete [] file_name;
            }
            if (export_bg_img) {
                if (!display_results)
                    BGS->GetBackgroundImage(bg_img);
                char* file_name = get_file_name(output_dir, "bg_img", "jpg", frame_idx);
                cvSaveImage(file_name, bg_img);
                delete [] file_name;
            }
			if (export_gradient_img&&GRD_used) {
				if (!display_results)
					BGS->GetCurrentGradientImage(cur_img_gradient);
				char* file_name = get_file_name(output_dir, "cur_img_gradient", "jpg", frame_idx);
				cvSaveImage(file_name, bg_img);
				delete [] file_name;
			}
            if (export_merged_img) {
                if (!display_results) {
                    if (!export_bg_img)
                        BGS->GetBackgroundImage(bg_img);
                    BGS->GetForegroundProbabilityImage(fg_prob_img3);
                    if (!export_fg_img)
                        BGS->GetForegroundImage(fg_img);
                    BGS->MergeImages(4, img, bg_img, fg_prob_img3, fg_img, merged_img);
                }
                char* file_name = get_file_name(output_dir, "merged_img", "jpg", frame_idx);
                cvSaveImage(file_name, merged_img);
                delete [] file_name;
            }

        }
#endif
    }
#ifdef BGSUB_LEARN
    /* save the learned background model */
    printf("\nSaving the background model: %s\n", bg_model_fn);
    BGS->Save(bg_model_fn, bg_model_save_type);
#endif
    /* release memories */
	if ( writer )
		cvReleaseVideoWriter(&writer);

	if (merged_img)
        cvReleaseImage(&merged_img);
    delete BGS;

    cvReleaseImage(&img);
    cvReleaseImage(&fg_img);
    cvReleaseImage(&fg_prob_img);
    cvReleaseImage(&fg_prob_img3);
    cvReleaseImage(&bg_img);
	//cvReleaseImage(&cur_img_gradient);
    if (LIST) delete LIST;
    if (CAM) delete CAM;

	printf("elapsed time: %lf",count_time_total);
    return 0;
}
