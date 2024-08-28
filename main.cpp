// OpenCV installing: https://docs.opencv.org/4.x/d7/d9f/tutorial_linux_install.html
// https://docs.opencv.org/4.x/d7/d9f/tutorial_linux_install.html


// cmake -S . -B ./build -DCMAKE_PREFIX_PATH=/home/svladykin/AIR/torchcpp/libtorch
// add to PATH /home/svladykin/AIR/torchcpp/libtorch/include/torch/csrc/api/include

// https://xtensor.readthedocs.io/en/latest/installation.html#from-source-with-cmake
// cmake -S . -B ./build
// cmake -DCMAKE_INSTALL_PREFIX=/home/svladykin/AIR/xtl/build/path_to_prefix --build ./build
// cd build && make install


//#define WITH_OPENCV
#include "matplotlibcpp.h"

#include <opencv2/opencv.hpp>

#include <torch/torch.h>
#include <torch/script.h>
#include <xtensor/xarray.hpp>
#include <xtensor/xio.hpp>
#include <xtensor/xview.hpp>


#include <iostream>
#include <getopt.h>
 
// using namespace cv;
namespace plt = matplotlibcpp;

#define IMAGE_SIZE  1024

class Detector {

     torch::Device device = torch::kCPU;
    
    uint16_t image_rows;
    uint16_t image_cols;
    
    
    std::string model_path;
    torch::jit::script::Module model;
    cv::Mat image;

    torch::jit::script::Module load_model(const char* model_path);
    torch::Tensor preprocess_image(cv::Mat img);
    cv::Mat process_model(torch::Tensor tensor_image);
    
    bool load_model()
    {
        if (std::ifstream(model_path)) {
            model = torch::jit::load(model_path, device);
            std::cout << "Loaded...." << std::endl;
        } else {
            std::cout << "Model " << model_path << " not exists" << std::endl;
            return false;
        }
        return true; 
    } 

public:
    Detector(std::string model_path) :  model_path(model_path) {}
    ~Detector() {};
    void run_model(cv::Mat original_image);

};

/* From segmentation_models_pytorch
   input_space="RGB"
   input_range=[0, 1]
   mean = [0.485, 0.456, 0.406]
   std = [0.229, 0.224, 0.225]
*/
cv::Mat preprocess_img_resnet50_imagenet(cv::Mat img)
{
    cv::Vec3d mean({0.485, 0.456, 0.406});
    cv::Vec3d std({0.229, 0.224, 0.225});

#if 1
    uchar test_data[3][2][3] = {{{200, 255, 10}, {50, 60, 70}},
                                 {{100, 130, 140}, {110, 80, 5}},
                                 {{30, 40, 50}, {80, 90, 100}}};

    cv::Mat test_matrix(3,2,CV_8UC3,test_data);

    std::cout << "test_matrix = " << test_matrix << "size = " << test_matrix.size() <<
        "channels = " << test_matrix.channels() << "total() = " << test_matrix.total()
        << "elemSize()" << test_matrix.elemSize() << std::endl << std::endl;
    
    
    uint8_t* pixelPtr = (uint8_t*)test_matrix.data;
    int cn = test_matrix.channels();
    //cv::Scalar_<uint8_t> bgrPixel;
    for(int i = 0; i < test_matrix.rows; i++)
    {
        for(int j = 0; j < test_matrix.cols; j++)
        {
            cv::Vec3b bgrPixel = test_matrix.at<cv::Vec3b>(i, j);
            //bgrPixel.val[0] = pixelPtr[i*test_matrix.cols*cn + j*cn + 0]; // B
            //bgrPixel.val[1] = pixelPtr[i*test_matrix.cols*cn + j*cn + 1]; // G
            //bgrPixel.val[2] = pixelPtr[i*test_matrix.cols*cn + j*cn + 2]; // R
            std::cout << bgrPixel; 
           // std::cout <<  (int)bgrPixel.val[0] << " " <<  (int)bgrPixel.val[1] << " " <<  (int)bgrPixel.val[2];       

        }
        std::cout << std::endl;
    }
    cv::Mat image = test_matrix;
#endif
//    cv::Mat image(img);
    image.convertTo(image, CV_64FC1, 1/255.0);

    std::cout << "AFTER convertTo CV_64FC1 = " << image << "size = " << image.size() <<
        "channels = " << image.channels() << "total() = "<< image.total()
        << "elemSize()" << image.elemSize() << std::endl << std::endl;
    
    //cv::normalize(image, image, 0, 1, cv::NORM_MINMAX, CV_32FC1);
    //image = image/255.0;

    std::cout << "AFTER image/255.0 m1 = " << image << "size = " << image.size <<
                    "channels() = " << image.channels() << std::endl << std::endl;

    for(int i = 0; i < image.rows; i++)
    {
        for(int j = 0; j < image.cols; j++)
        {
            cv::Vec3d bgrPixel = image.at<cv::Vec3d>(i, j);
            //bgrPixel.val[0] = pixelPtr[i*test_matrix.cols*cn + j*cn + 0]; // B
            //bgrPixel.val[1] = pixelPtr[i*test_matrix.cols*cn + j*cn + 1]; // G
            //bgrPixel.val[2] = pixelPtr[i*test_matrix.cols*cn + j*cn + 2]; // R
            std::cout << bgrPixel << " === "; 
            std::cout << (double)image.at<cv::Vec3d>(i, j).val[0] << " " << 
                         (double)image.at<cv::Vec3d>(i, j).val[1] << " " <<  
                         (double)image.at<cv::Vec3d>(i, j).val[2] << "   ";
        }
        std::cout << std::endl;
    }

    std::cout << "AFTER NORMALIZATION:" << std::endl;
    for (size_t i = 0; i < image.rows; i++)
    {
        for (size_t j = 0; j < image.cols; j++)
        {
            image.at<cv::Vec3d>(i, j).val[0] -= mean[0];
            image.at<cv::Vec3d>(i, j).val[1] -= mean[1];
            image.at<cv::Vec3d>(i, j).val[2] -= mean[2];
            image.at<cv::Vec3d>(i, j).val[0] /= std[0];
            image.at<cv::Vec3d>(i, j).val[1] /= std[1];
            image.at<cv::Vec3d>(i, j).val[2] /= std[2];

  //          std::cout << image.at<cv::Vec3d>(i, j) << " === "; 
  //          std::cout << (double)image.at<cv::Vec3d>(i, j).val[0] << " " << 
  //                       (double)image.at<cv::Vec3d>(i, j).val[1] << " " <<  
  //                       (double)image.at<cv::Vec3d>(i, j).val[2] << "   ";
        }
        std::cout << std::endl;
    }


    std::cout << "AFTER PREPROCESSING image = " << image <<
        " size = " << image.size() <<
        " channels = " << image.channels() << " total() = " << image.total()
        << " elemSize()" << image.elemSize() << std::endl << std::endl;

 //   cv::imshow("Preprocessed image", image);
 //   cv::waitKey(0);
    
    exit(0);
    return image;
}


torch::Tensor Detector::preprocess_image(cv::Mat image) 
{
    if (image.empty())
    {
        std::cout << "Unable to read image" << std::endl;
        exit(0);
    }
#if 0
    //cv::namedWindow("Input image", cv::WINDOW_NORMAL);
    cv::imshow("Source image", image);
    cv::waitKey(0);

    uint8_t test_array[] = {255, 255, 255,   0,   0,   0, 255, 255, 255,
                                            255, 255, 255,  255, 255, 255,   0,   0,   0,
                                            0,   0,   0, 255, 255, 255,   0,   0,   0};

    cv::Mat test(3, 3, CV_8UC3, test_array);


        std::cout << "test image 3 X 3" << test << " img.size()  = " << 
          test.size() << "img.channels() = " << test.channels() << std::endl << std::endl;

        cv::waitKey(0);

        cv::resize(test, test, cv::Size(IMAGE_SIZE, IMAGE_SIZE));


    cv::imshow("resizedImage", test);

    cv::waitKey(0);
#endif

    image_rows = image.rows;
    image_rows = image.cols;

    cv::resize(image, image, cv::Size(IMAGE_SIZE, IMAGE_SIZE));
    cv::cvtColor(image, image, cv::COLOR_BGR2RGB);
    //cv::imshow("colorConvertedImage", image);
    //cv::waitKey(0);

    // normalization???
    //img = preprocessing_fn(img);
   // image = preprocess_img_resnet50_imagenet(image);

  //  cv::imshow("Preprocessed image", image);
  //  cv::waitKey(0);


#if 0
    torch::Tensor tensor_image = torch::from_blob(image.data, 
                                        {1, image.rows, image.cols, image.channels() }, 
                                        torch::kByte);
#endif
#if 0
    //http://note4lin.top/post/pytorch%E9%83%A8%E7%BD%B2/
    torch::Tensor tensor_image = torch::from_blob(image.data, 
                                        {image.rows, image.cols,3},
                                        torch::kByte);


    // x_tensor = to_tensor(img) in main.py, place the RGB channels first
    // --- tensor_image = tensor_image.permute({2,0,1});  
    tensor_image = tensor_image.permute({2,0,1});  

    tensor_image = tensor_image.toType(torch::kFloat);
//    tensor_image = tensor_image.div(255);  // black screen???
    tensor_image = tensor_image.to(torch::kCPU);
    tensor_image = tensor_image.unsqueeze(0);
#else
  std::vector<double> mean = {0.406, 0.456, 0.485};
  std::vector<double> std = {0.225, 0.224, 0.229};
  //cv::resize(img, img, cv::Size(IMG_SIZE, IMG_SIZE));
  cv::Mat img_copy = image;
  image.convertTo(image, CV_32FC3, 1.0f / 255.0f);
  torch::Tensor frame_tensor =
      torch::from_blob(image.data, {1, image.rows, image.cols, image.channels() /* 3*/}); 
  
  frame_tensor = frame_tensor.permute({0, 3, 1, 2});
  frame_tensor = torch::data::transforms::Normalize<>(mean, std)(frame_tensor);
  frame_tensor = frame_tensor.to(torch::kCPU);
#endif 
    // from here
    //     img_tensor = torch::data::transforms::Normalize<>(norm_mean, norm_std)(img_tensor);
    return frame_tensor;
}

cv::Mat Detector::process_model(torch::Tensor tensor_image)
{
    cv::Mat img;
    std::vector<torch::jit::IValue> inputs;
    inputs.push_back(tensor_image);

    torch::Tensor tensor_pred_mask = model.forward(inputs).toTensor().to(torch::kCPU);
#if 0
    double test_data[1][2][3][3] = {{{{0.5481, 0.7580, 0.1392},
          {0.2583, 0.1031, 0.7573},
          {0.2903, 0.6812, 0.9128}},

         {{0.6619, 0.0153, 0.8344},
          {0.8775, 0.4836, 0.5601},
          {0.2078, 0.8901, 0.6613}}}};

    torch::Tensor tensor_pred_mask = torch::from_blob(test_data,{1,2,3,3},torch::kDouble);
    tensor_pred_mask = torch::rand({1, 2, 3, 3}, torch::requires_grad());
#endif

    std::cout << "AFTER forward tensor " << /*tensor_pred_mask <<*/ " sizes = " <<  tensor_pred_mask.sizes() << std::endl << std::endl;
  

    tensor_pred_mask = tensor_pred_mask.to(torch::kCPU);   // is it neccessary ???
    tensor_pred_mask = tensor_pred_mask.squeeze(0);
    tensor_pred_mask.detach();

    std::cout << "AFTER squeeze(0) tensor " <</* tensor_pred_mask << */" sizes = " << tensor_pred_mask.sizes() 
                                            << std::endl << std::endl;


    tensor_pred_mask = tensor_pred_mask.permute({1, 2, 0});
    std::cout << "AFTER permute(1, 2, 0) tensor " <</* tensor_pred_mask << */" sizes = " << tensor_pred_mask.sizes() 
                        << std::endl << std::endl;

    // given segmentation , we can do argmax(0) without tensor permuting
    torch::Tensor reverse_one_hot = tensor_pred_mask.argmax(2).toType(torch::kU8);
    std::cout << "AFTER reverse_one_hot tensor " /*<< reverse_one_hot */<< " sizes = " << 
            reverse_one_hot.sizes() << std::endl << std::endl;

    // CV_8UC3 is an 8-bit unsigned integer matrix/image with 3 channels
    img = cv::Mat(cv::Size{ 1024, 1024}, CV_8UC1, reverse_one_hot.data_ptr());

    std::cout << "AFTER conversion to image " /*<< img */<< " img.size()  = " << 
          img.size() << "img.channels() = " << img.channels() << std::endl << std::endl;

    //img.convertTo(img, CV_8UC3);

    //std::cout << "AFTER conversion to CV_8UC3 " << img << " img.size()  = " << 
    //      img.size() << "img.channels() = " << img.channels() << std::endl << std::endl;

    cvtColor(img, img, cv::COLOR_GRAY2RGB);
    std::cout << "AFTER conversion to COLOR_GRAY2RGB " /*<< img*/ << " img.size()  = " << 
            img.size() << "img.channels() = " << img.channels() << std::endl << std::endl;

    // maxarg index 0/1to RGB BLACK/WHITE
    img = img * 255;

    std::cout << "AFTER img * 255 " /*<< img*/ << " img.size()  = " << 
          img.size() << "img.channels() = " << img.channels() << std::endl << std::endl;

    cv::imwrite("new_img.jpg", img); 
    
    // to rebuild opencv from sources to make this work
    //cv::imshow("FinishImage", img);
//    PyObject* mat;
//    plt::imshow(img.ptr<uchar>(), img.cols, img.rows, 3, {}, &mat);
//    plt::show();

 //   cv::waitKey(0);




    std::cout << "Tensor converted to Image!!" << std::endl;

    inputs.clear();  
    return img;
}


void Detector::run_model(cv::Mat original_image)
{
    std::cout << "Model loading..." << std::endl;

    load_model();

    std::cout << "Loaded...." << std::endl;

    std::cout << "Image preprocessing..." << std::endl;
    torch::Tensor tensor_image = preprocess_image(original_image);
    std::cout << "Done..." << std::endl;

    std::cout << "Model processing..." << std::endl;
    auto start_time = std::chrono::high_resolution_clock::now();
    cv::Mat pred_mask = process_model(tensor_image);
    auto end_time = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end_time - start_time);
    std::cout << "Process time: " << duration.count() << " milliseconds" << std::endl;

    std::cout << "Done." << std::endl;


    //cv::resize(pred_mask, pred_mask, cv::Size(1024, 1024));
    //plt::plot(x, y);
    //namedWindow("Result:", WINDOW_NORMAL);
    //imshow("Result", pred_mask);


}

cv::Mat load_image(const char* img_path) 
{
    cv::Mat image;
    if (std::ifstream(img_path)) {
        image = cv::imread(img_path, cv::IMREAD_COLOR);
    } else {
        std::cout << "File " << img_path << " not exists" << std::endl;
        exit(-1);
    }
    return image;
}

     
int main(int argc, char *argv[])
{
	const char* usage =
        "Usage\n\n";
	const char* short_options = "hmi";
    const struct option long_options[] = {
    		{ .name = "help",       .has_arg=no_argument,       .flag=NULL, .val='h' },
            { .name = "model",      .has_arg=required_argument, .flag=NULL, .val='m' },
            { .name = "image",      .has_arg=required_argument, .flag=NULL, .val='i' },
            { .name = NULL,         .has_arg=0,                 .flag=NULL, .val=0 }
        };
        
    int rez;
    
    const char* model_file_name = "model/model_traced.torchscript";
    const char* image_file_name = "image2.jpg";

    while ((rez=getopt_long(argc,argv,short_options, long_options, NULL))!=-1)
    {
        switch (rez) 
        {
            case 'h':
                std::cout << usage;
                exit(0);
            case 'm':
            	model_file_name = optarg;
            	break;
            case 'i':
                image_file_name = optarg;
                break;
           }
    }
    
    std::cout << "model: " << model_file_name << std::endl;
    std::cout << "image: " << image_file_name << std::endl;
        

    Detector detector(model_file_name);
    cv::Mat image = load_image(image_file_name);
    detector.run_model(image);

    return 0;

}
