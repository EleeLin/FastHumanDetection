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
// LocalBinaryPattern.h: interface for the CLocalBinaryPattern class.
//
//////////////////////////////////////////////////////////////////////

#if !defined(_LOCAL_BINARY_PATTERN_H_)
#define _LOCAL_BINARY_PATTERN_H_

#include "cv.h"
#include "BGS.h"


/************************************************************************/
/* two types of computing the LBP operators but currently GENERAL_LBP   */
/* has been implemented.                                                */
/************************************************************************/
#define	GENERAL_LBP	0
#define SYMMETRIC_LBP	1

#include <cstdio>						// C I/O (for sscanf)
#include "OpencvDataConversion.h"


class CLocalBinaryPattern  
{
public:
    void CalImageDifferenceMapMask(IplImage *cent_img, IplImage *neig_img, float *pattern, float lbp_level_weight, uchar *mask,CvRect *roi=NULL);
	void CalImageDifferenceMap(IplImage *cent_img, IplImage *neig_img, float *pattern, float lbp_level_weight, CvRect *roi=NULL);
	void CalNeigPixelOffset(float radius, int tot_neig_pts_num, int neig_pt_idx, int &offset_x, int &offset_y);
	void CalShiftedImage(IplImage *src, int offset_x, int offset_y, IplImage *dst, CvRect *roi=NULL);
	void FreeMemories();
	void ComputeLBP(PixelLBPStruct *PLBP, CvRect *roi=NULL);
	void ComputeLBPMask(PixelLBPStruct *PLBP, uchar *mask, CvRect *roi = NULL);
	void SetNewImages(IplImage **new_imgs);
	
	IplImage** m_ppOrgImgs;			/* the original images used for computing the LBP operators */

	void Initialization(IplImage **first_imgs, int imgs_num, 
			int level_num, float *radius, int *neig_pt_num, 
			float robust_white_noise = 3.0f, int type = GENERAL_LBP, bool ltp_used = false);
	
	CLocalBinaryPattern();
	virtual ~CLocalBinaryPattern();

	float	m_fRobustWhiteNoise;		/* the robust noise value for computing the LBP operator in each channel */
	bool    m_bLTPUsed;                 /* use the LTP for texture description */

private:
	void SetShiftedMeshGrid(CvSize img_size, float offset_x, float offset_y, CvMat *grid_map_x, CvMat *grid_map_y);

	float*	m_pRadiuses;			/* the circle radiuses for the LBP operator */
	int	m_nLBPType;			/* the type of computing LBP operator */
	int*	m_pNeigPointsNums;		/* the numbers of neighboring pixels on multi-level circles */
	int	m_nImgsNum;			/* the number of multi-channel image */
	int	m_nLBPLevelNum;			/* the number of multi-level LBP operator */
	CvSize	m_cvImgSize;			/* the image size (width, height) */

	CvPoint* m_pXYShifts;
	CvPoint	m_nMaxShift;

	IplImage* m_pShiftedImg;

public:
	float m_f_lbp_level_weight_exponential_decaying_constant;    /* the exponential decaying constant of the multi-level lbp */
};

#endif // !defined(_LOCAL_BINARY_PATTERN_H_)

