
#include "ov2311_uc593c.hpp"
#include <core/iController.hpp>
#include <spdlog/spdlog.h>

#include <opencv2/opencv.hpp>
#include <thread>
#include <time.h>
#include <iostream>
#include <istream>
#include <string>
#include <sys/types.h> 
#include <sys/stat.h> 
#include <chrono>

using namespace std;


static bool _running = true;
static ArduCamHandle _handle;  //camera handle
static ArduCamCfg _config; //camera configurations

namespace ppe::cmos {

    /* image capture thread function */
    void captureImage_thread(ArduCamHandle handle) {
        Uint32 rtn_val = ArduCam_beginCaptureImage(_handle);
        if(rtn_val==USB_CAMERA_USB_TASK_ERROR) {
            spdlog::error("Error beginning capture({})", rtn_val);
            return;
        }

        while(_running){
            rtn_val = ArduCam_captureImage(_handle);
            if(rtn_val==USB_CAMERA_USB_TASK_ERROR) {
                spdlog::error("Error beginning capture image({})", rtn_val);
                break;
            }
            else if(rtn_val>0xff){
                spdlog::error("Error beginning capture image({})", rtn_val);
            }
        }
        _running = false;
        ArduCam_endCaptureImage(_handle);
    }

    ov2311_uc593c::ov2311_uc593c(const char* config, unsigned int max_fps):_config_file(config), _max_fps(max_fps) {
        camera_matrix = (cv::Mat1d(3,3) << _FX_, 0., _CX_, 0., _FY_, _CY_, 0., 0., 1.);
        distortion_coeff = (cv::Mat1d(1,5) << _K1_, _K2_, 0, _P1_, _P2_);
    }
    
    ov2311_uc593c::~ov2311_uc593c(){
        _running = false;
    }

    bool ov2311_uc593c::open(){

        //1. Scan camera module
        ArduCamIndexinfo _camInfo[16];
        unsigned int n_cam = ArduCam_scan(_camInfo);
        spdlog::info("{} Camera Found", n_cam);

        if(n_cam==0){
            spdlog::error("Camera does not exist");
            return false;
        }

        this_thread::sleep_for(std::chrono::milliseconds(2000)); //wait for 2 sec, hardware bug?

        //2. read camera configuration from file
        if(this->configure(_config_file.c_str(), _handle, _config)){
            ArduCam_setMode(_handle, CONTINUOUS_MODE);
            _capture_thread = new thread(captureImage_thread, _handle);
            return true;
        }
        return false;
    }

    void ov2311_uc593c::close(){
        spdlog::info("Closed camera device");
        _running = false;
		_capture_thread->join();
        ArduCam_close(_handle);
    }

    bool ov2311_uc593c::is_valid(){
        return true;
    }

    cv::Mat ov2311_uc593c::convert(ArduCamOutData* frameData){
        cv::Mat rawImage;
        Uint8* data = frameData->pu8ImageData;
        this->width = _config.u32Width;
        this->height = _config.u32Height;

        switch (_config.emImageFmtMode){
            case FORMAT_MODE_RGB:
                spdlog::info("FORMAT MODE : FORMAT_MODE_RGB");
                //rawImage = RGB565toMat(data, width, height);
                break;
            case FORMAT_MODE_RAW_D:
                spdlog::info("FORMAT MODE : FORMAT_MODE_RAW_D");
                /*rawImage = separationImage(data, width, height);
                switch (color_mode) {
                case RAW_RG:cv::cvtColor(rawImage, rawImage, cv::COLOR_BayerRG2BGR);
                    break;
                case RAW_GR:cv::cvtColor(rawImage, rawImage, cv::COLOR_BayerGR2BGR);
                    break;
                case RAW_GB:cv::cvtColor(rawImage, rawImage, cv::COLOR_BayerGB2BGR);
                    break;
                case RAW_BG:cv::cvtColor(rawImage, rawImage, cv::COLOR_BayerBG2BGR);
                    break;
                default:
                    cv::cvtColor(rawImage, rawImage, cv::COLOR_BayerRG2BGR);
                    break;
                }*/
                break;
            case FORMAT_MODE_MON_D:
                spdlog::info("FORMAT MODE : FORMAT_MODE_MON_D");
                //rawImage = separationImage(data, width, height);
                break;
            case FORMAT_MODE_JPG:
                spdlog::info("FORMAT MODE : FORMAT_MODE_JPG");
                //rawImage = JPGToMat(data, frameData->stImagePara.u32Size);
                break;
            case FORMAT_MODE_RAW:
                if (_config.u8PixelBytes == 2) {
                    rawImage = dBytesToMat(data, frameData->stImagePara.u8PixelBits, this->width, this->height);
                }
                else {
                    rawImage = BytestoMat(data, this->width, this->height);
                }
                break;
            case FORMAT_MODE_YUV:
                spdlog::info("FORMAT MODE : FORMAT_MODE_YUV");
                //rawImage = YUV422toMat(data, width, height);
                break;
            case FORMAT_MODE_MON:
                spdlog::info("FORMAT MODE : FORMAT_MODE_MON");
                /*if (cameraCfg.u8PixelBytes == 2) {
                    rawImage = dBytesToMat(data, frameData->stImagePara.u8PixelBits, width, height);
                }
                else {
                    rawImage = BytestoMat(data, width, height);
                }*/
                break;
            default:
                spdlog::info("FORMAT MODE : DEFAULT");
                /*if (cameraCfg.u8PixelBytes == 2) {
                    rawImage = dBytesToMat(data, frameData->stImagePara.u8PixelBits, width, height);
                }
                else {
                    rawImage = BytestoMat(data, width, height);
                }
                cv::cvtColor(rawImage, rawImage, cv::COLOR_BayerRG2RGB);*/
                break;
            }

        return rawImage;
    }
    

    cv::Mat ov2311_uc593c::capture(){
        if(ArduCam_availableImage(_handle)>0){
            ArduCamOutData* frameData;
            Uint32 rtn_val = ArduCam_readImage(_handle, frameData);
            if(rtn_val==USB_CAMERA_NO_ERROR){
                cv::Mat raw = convert(frameData);
                if(raw.empty()){
                    spdlog::error("Image FIFO is empty");
                }
                else {
                    ArduCam_del(_handle);
                    return raw;
                }
            }
        }
        return cv::Mat();
    }

    void ov2311_uc593c::config_board(ArduCamHandle &cameraHandle, Config config){
        uint8_t u8Buf[10];
        for (unsigned int n = 0; n < config.params[3]; n++) {
            u8Buf[n] = config.params[4 + n];
        }
        ArduCam_setboardConfig(cameraHandle, config.params[0], config.params[1], config.params[2], config.params[3], u8Buf);
    }

    bool ov2311_uc593c::configure(const char* filename, ArduCamHandle &handle, ArduCamCfg &config){
        CameraConfigs cfgs;
        memset(&cfgs, 0x00, sizeof(CameraConfigs));

        //1. read config file parse
        int parse_res = arducam_parse_config(filename, &cfgs);
        if(parse_res) {
            spdlog::error("Cannot find configuration file : {}", filename);
            return false;
        }

        // set I2C mode
        spdlog::info("Set camera config I2C Mode : {}", cfgs.camera_param.i2c_mode);
        switch(cfgs.camera_param.i2c_mode){
            case 0: _config.emI2cMode = I2C_MODE_8_8; break;
            case 1: _config.emI2cMode = I2C_MODE_8_16; break;
            case 2: _config.emI2cMode = I2C_MODE_16_8; break;
            case 3: _config.emI2cMode = I2C_MODE_16_16; break;
            default: break;
        }

        // set Color format
        //int color_mode = (cfgs.camera_param.format&0xff);
        uint16_t format_mode = cfgs.camera_param.format>>8;
        spdlog::info("Set camera config image format mode : {}", format_mode);
        switch(format_mode){
            case 0: _config.emImageFmtMode = FORMAT_MODE_RAW; break;
            case 1: _config.emImageFmtMode = FORMAT_MODE_RGB; break;
            case 2: _config.emImageFmtMode = FORMAT_MODE_YUV; break;
            case 3: _config.emImageFmtMode = FORMAT_MODE_JPG; break;
            case 4: _config.emImageFmtMode = FORMAT_MODE_MON; break;
            case 5: _config.emImageFmtMode = FORMAT_MODE_RAW_D; break;
            case 6: _config.emImageFmtMode = FORMAT_MODE_MON_D; break;
            default: break;
        }

        // set other parameters
        _config.u32Width = cfgs.camera_param.width;
        _config.u32Height = cfgs.camera_param.height;
        _config.u32I2cAddr = cfgs.camera_param.i2c_addr;
        _config.u8PixelBits = cfgs.camera_param.bit_width;
        _config.u32TransLvl = cfgs.camera_param.trans_lvl;
        spdlog::info("Set image size : {}x{}", cfgs.camera_param.width, cfgs.camera_param.height);
        spdlog::info("Set I2C Address : {}", cfgs.camera_param.i2c_addr);
        spdlog::info("Set Bit width : {}", cfgs.camera_param.bit_width);
        spdlog::info("Set Trans level : {}", cfgs.camera_param.trans_lvl);

        if(_config.u8PixelBits<=8){ _config.u8PixelBytes = 1; }
        else if(_config.u8PixelBits>8 && _config.u8PixelBits<=16){
            _config.u8PixelBytes = 2;
            //save_raw = true;
        }

        //2. open
        int res = ArduCam_autoopen(_handle, &_config);
        if(res==USB_CAMERA_NO_ERROR){
            for(unsigned int i=0; i<cfgs.configs_length; i++){
                uint32_t type = cfgs.configs[i].type;
                if (((type >> 16) & 0xff) && ((type >> 16) & 0xff) != config.usbType)
                    continue;
                switch (type & 0xffff) {
                case CONFIG_TYPE_REG: { ArduCam_writeSensorReg(handle, cfgs.configs[i].params[0], cfgs.configs[i].params[1]); } break;
                case CONFIG_TYPE_DELAY: { usleep(1000*cfgs.configs[i].params[0]); } break;
                case CONFIG_TYPE_VRCMD: { config_board(handle, cfgs.configs[i]); } break;
                }
            }

            ArduCam_registerCtrls(handle, cfgs.controls, cfgs.controls_length);
        }
        else {
            switch(res){
                case USB_CAMERA_USB_CREATE_ERROR: spdlog::error("USB create error"); break;
                case USB_CAMERA_USB_SET_CONTEXT_ERROR: spdlog::error("USB set context error"); break;
                case USB_CAMERA_VR_COMMAND_ERROR: spdlog::error("USB Vendor command error"); break;
                case USB_CAMERA_USB_VERSION_ERROR: spdlog::error("USB version error"); break;
                case USB_CAMERA_BUFFER_ERROR: spdlog::error("Camera buffer error"); break;
                case USB_CAMERA_NOT_FOUND_DEVICE_ERROR: spdlog::error("Camera not found device error"); break;
            }
            return false;
        }

        // set framerate
        ArduCam_setCtrl(handle, "setFramerate", (Int64)_max_fps);

        return true;
    }

    cv::Mat ov2311_uc593c::dBytesToMat(Uint8* bytes, int bit_width, int width, int height) {
        unsigned char* temp_data = (unsigned char*)malloc(width * height);
        int index = 0;
        for (int i = 0; i < width * height * 2; i += 2) {
            unsigned char temp = ((bytes[i + 1] << 8 | bytes[i]) >> (bit_width - 8)) & 0xFF;
            temp_data[index++] = temp;
        }
        cv::Mat image = cv::Mat(height, width, CV_8UC1);
        memcpy(image.data, temp_data, _config.u32Height * _config.u32Width);
        free(temp_data);
        return image;
    }

    cv::Mat ov2311_uc593c::BytestoMat(Uint8* bytes, int width, int height)
    {
        cv::Mat image = cv::Mat(height, width, CV_8UC1, bytes);
        return image;
    }

}