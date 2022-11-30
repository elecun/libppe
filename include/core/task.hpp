/**
 * @file task.hpp
 * @author Byunghun Hwang(bh.hwagn@iae.re.kr)
 * @brief task base class
 * @version 0.1
 * @date 2022-01-25
 * 
 * @copyright Copyright (c) 2022
 * 
 */

#ifndef _CORE_TASK_HPP_
#define _CORE_TASK_HPP_

#include <thread>
#include <vector>
#include <string>
#include <map>
#include <opencv2/core.hpp>
#include <include/spdlog/spdlog.h>

using namespace std;

namespace ppe {

    class task {
        public:
            enum class STATUS : int {
                STOPPED,
                RUNNING
            };

            friend class taskmanager;

            task() { }
            virtual ~task() { }

            /**
             * @brief start task with thread
             * 
             * @return true if success
             * @return false 
             */
            virtual bool start() = 0;

            /**
             * @brief stop task with thread
             * 
             * @return true if success
             * @return false 
             */
            virtual bool stop() = 0;

            /**
             * @brief Get the status object
             * 
             * @return task::STATUS 
             */
            virtual task::STATUS get_status(){
                return _status;
            }

        protected:
            virtual void set_status(task::STATUS status){
                _status = status;
            }

            virtual void process(cv::Mat data) = 0;
            virtual void request(cv::Mat data) = 0;
            virtual void wait() = 0;

        protected:
            thread* _thread = nullptr;
            pthread_t _handle;
            condition_variable _cv;
            mutex _lock;

        private:
            task::STATUS _status = task::STATUS::STOPPED;

    }; 

    class taskmanager {
        public:
            taskmanager() = default;
            virtual ~taskmanager(){
                for(auto itr=_task_container.begin();itr!=_task_container.end();++itr){
                    itr->second->stop();
                    delete itr->second;
                }
                _task_container.clear();
            }

            /**
             * @brief start all task or specific task
             * 
             * @param taskname 
             * @return true 
             * @return false 
             */
            virtual bool start(const char* taskname = nullptr){
                if(_task_container.empty())
                    return false;

                if(taskname!=nullptr){
                    map<string, task*>::iterator itr = _task_container.find(taskname);
                    if(itr!=_task_container.end()){
                        if(itr->second->get_status()==task::STATUS::STOPPED)
                            itr->second->start();
                        return true;
                    }
                }
                else {
                    for(auto itr=_task_container.begin();itr!=_task_container.end(); ++itr){
                        if(itr->second->get_status()==task::STATUS::STOPPED)
                            itr->second->start();
                    }
                    return true;
                }

                return false;
            }

            /**
             * @brief stop all or specific task
             * 
             * @param taskname 
             * @return true 
             * @return false 
             */
            virtual bool stop(const char* taskname = nullptr){
                if(_task_container.empty())
                    return false;

                if(taskname!=nullptr){
                    map<string, task*>::iterator itr = _task_container.find(taskname);
                    if(itr!=_task_container.end()){
                        if(itr->second->get_status()==task::STATUS::RUNNING)
                            itr->second->stop();
                        return true;
                    }
                }
                else {
                    for(auto itr=_task_container.begin();itr!=_task_container.end(); ++itr){
                        if(itr->second->get_status()==task::STATUS::RUNNING)
                            itr->second->stop();
                    }
                    return true;
                }
                return false;
            }
            
            /**
             * @brief add task into task container
             * 
             * @param taskname taskname
             * @param instance task instance pointer
             * @return true 
             * @return false 
             */
            virtual bool add(const char* taskname, task* instance){
                if(_task_container.find(taskname)==_task_container.end()){
                    _task_container.insert(pair<string, task*>(taskname, instance));
                    return true;
                }
                return false;
            }
            
            /**
             * @brief remove task in task container
             * 
             * @param taskname 
             * @return true 
             * @return false 
             */
            virtual bool remove(const char* taskname){
                if(_task_container.find(taskname)!=_task_container.end()){
                    return false;
                }
                delete _task_container[taskname];
                _task_container.erase(taskname);
                return true;
            }

            /**
             * @brief data process request to each task processing
             * 
             * @param data image data
             */
            virtual void request_all(cv::Mat data){
                for(auto itr=_task_container.begin();itr!=_task_container.end(); ++itr){
                    if(itr->second->get_status()==task::STATUS::RUNNING)
                        itr->second->request(data);
                }
            }

            virtual void request(ppe::task* t, cv::Mat data){
                
            }

            void wait(){
                for(auto itr=_task_container.begin();itr!=_task_container.end(); ++itr){
                    if(itr->second->get_status()==task::STATUS::RUNNING){
                        itr->second->wait();
                    }
                        
                }
            }

        private:
            map<string, task*> _task_container;
    };
} /* namespace */

#endif