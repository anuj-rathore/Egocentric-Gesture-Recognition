#include <opencv2/opencv.hpp>

#include "FeatureComputer.hpp"
#include "Classifier.h"
#include "LcBasic.h"
#include "HandDetector.hpp"
#include <iostream>
#include <cstdio>

using namespace std;
using namespace cv;

int main (int argc, char * const argv[])
{
    bool TRAIN_MODEL = 0;           //1 if you are training the models, 0 if you are running the program to predict
    bool TEST_MODEL  = 1;           //0 if you are training the models, 1 if you are running the program to predict
    
    int target_width = 320;			// for resizing the input (small is faster)
    
    // maximum number of image masks that you will use
    // must have the masks prepared in advance
    // only used at training time
    int num_models_to_train = 442;
  
    
    // number of models used to compute a single pixel response
    // must be less than the number of training models
    // only used at test time
    int num_models_to_average = 10;
    
    // runs detector on every 'step_size' pixels
    // only used at test time
    // bigger means faster but you lose resolution
    // you need post-processing to get contours
    int step_size = 3;
    
    // Assumes a certain file structure e.g., /root/img/basename/00000000.jpg
    string root = "/home/anuj/Ego-Centric-Gesture-Recognition/";       //replace with path to your Xcode project
    string basename = "";
    string img_prefix		= root + "img"		+ basename + "/";			// color images
    string msk_prefix		= root + "mask"     + basename + "/";			// binary masks
    string model_prefix		= root + "models"	+ basename + "/";			// output path for learned models
    string globfeat_prefix  = root + "globfeat" + basename + "/";			// output path for color histograms
    
    
    // types of features to use (you will over-fit if you do not have enough data)
    // r: RGB (5x5 patch)
    // v: HSV
    // l: LAB
    // b: BRIEF descriptor
    // o: ORB descriptor
    // s: SIFT descriptor
    // u: SURF descriptor
    // h: HOG descriptor
    string feature_set = "rvl";
    
    
    
    if(TRAIN_MODEL)
    {
        cout << "Training...\n";
        HandDetector hd;
        cout <<"check"<<msk_prefix<<endl;
        hd.loadMaskFilenames(msk_prefix);
        hd.trainModels(basename, img_prefix, msk_prefix,model_prefix,globfeat_prefix,feature_set,num_models_to_train,target_width);
        cout << "Done Training...\n";
    }
    
    for (int i=1;i<=7;i++){
        if(TEST_MODEL)
        {
            cout << "Testing...\n";
            string vid_filename		= root + "vid/"+ basename + ".avi";
            
            HandDetector hd;
            hd.testInitialize(model_prefix,globfeat_prefix,feature_set,num_models_to_average,target_width);
            
            // VideoCapture cap(0);
            
            VideoCapture cap(root + "data2/slide_1_0" + to_string(i) + ".avi"); // open the video file for reading
            if ( !cap.isOpened() )  // if not success, exit program
            {
                 cout << "Cannot open the video file" << endl;
                 return -1;
            }

            Mat ppr;
            
            VideoWriter avi;

            int frame_width= cap.get(CV_CAP_PROP_FRAME_WIDTH)/2 ;
            int frame_height = cap.get(CV_CAP_PROP_FRAME_HEIGHT)/2;
            VideoWriter video("/home/anuj/Ego-Centric-Gesture-Recognition/out2/slide_1_0" + to_string(i) + ".avi",CV_FOURCC('M','J','P','G'),25, Size(frame_width,frame_height),true);

            
            while(1)
            {
                Mat im;
                cap >> im;
                cout<<im.rows<<" "<<im.cols<<endl;
                cout << "frame height" << frame_height<< endl;
                cout << "frame width" << frame_width << endl;  
                // video.write(im);
                
                if(!im.data) break;
                //cap >> im; if(!im.data) break; // skip frames with these
                //cap >> im; if(!im.data) break;
                //cap >> im; if(!im.data) break;
                
                hd.test(im,num_models_to_average,step_size);
                
                
                // Different ways to visualize the results
                // hd._response_img (float probabilities in a matrix)
                // hd._blur (blurred version of _response_img)
                
                
                int SHOW_RAW_PROBABILITY = 1;
                if(SHOW_RAW_PROBABILITY)
                {
                    Mat raw_prob;
                    hd.colormap(hd._response_img,raw_prob,0);
                    imshow("probability",raw_prob);	// color map of probability
                }
                
                int SHOW_BLUR_PROBABILITY = 1;
                if(SHOW_BLUR_PROBABILITY)
                {
                    Mat pp_res;
                    hd.postprocess(hd._response_img);
                    imshow("blurred",hd._blu);		// colormap of blurred probability
                }
                

                    Mat pp_contour = hd.postprocess(hd._response_img);		// binary contour
                    cout<<pp_contour.rows<<" "<<pp_contour.cols<<endl;
                   
                    hd.colormap(pp_contour,pp_contour,0);	
                    cout<<pp_contour.rows<<" "<<pp_contour.cols<<endl;
                   				// colormap of contour
                    cout<<"hey"<<endl;
                    video.write(pp_contour);
                    imshow("contour",pp_contour);
                    
                
                
                int SHOW_RES_ALPHA_BLEND = 1;
                if(SHOW_RES_ALPHA_BLEND)
                {
                    Mat pp_res = hd.postprocess(hd._response_img);
                    hd.colormap(pp_res,pp_res,0);
                    resize(pp_res,pp_res,im.size(),0,0,INTER_LINEAR);
                    addWeighted(im,0.7,pp_res,0.3,0,pp_res);				// alpha blend of image and binary contour
                    imshow("alpha_res",pp_res);                
                    
                }
                			
                 // if(!avi.isOpened())
                 // {
                 // stringstream ss;
                 // ss.str("");
                 // ss << root + "/vis/" + basename + "_skin.avi"; 
                 // int fourcc = avi.fourcc('F','L','V','1');
                 // avi.open(ss.str(),fourcc,30,ppr.size(),true);
                 // }
                 // avi << ppr;
                             
                waitKey(1);
            }
            video.release();
        }
    }
}