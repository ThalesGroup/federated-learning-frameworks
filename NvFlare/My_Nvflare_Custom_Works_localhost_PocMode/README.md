- This project represents some custom work with Nvfalre in POC mode running on localhost (you can change the port and IP address of all components to run them on different machines, see the [NVFlare_Remote_FL](nvflare_remote_fl) folder for more details).

- We are using 2 clients and a server.

- Custom implementation code of all jobs can be found in /poc/admin/transfer/ folder.

- Launch a container with NvFlare installed (see the nvflare_remote_fl folder to build this image)

      docker run -d -it --name nvflare_poc -v ./poc:/workspace/poc  nvflare-image

- Open 4 terminals:

  - run server:

        docker exec -it nvflare_poc bash
        cd /workspace/poc/server/startup
        ./start.sh

  - run client1:

        docker exec -it nvflare_poc bash
        cd /workspace/poc/site-1/startup
        ./start.sh

  - run client2:

        docker exec -it nvflare_poc bash
        cd /workspace/poc/site-2/startup
        ./start.sh

  - run admin: 

        docker exec -it nvflare_poc bash
        cd /workspace/poc/admin/startup
        ./fl_admin.sh

  - submit job in admin side (ex. IDD_MLP_Tensorflow job):

        submit_job /workspace/poc/admin/transfer/Mnist_Pytorch_MLP_IID/jobs/Mnist_Pytorch_MLP_IID

  - Verify the status of the clients by running

        check_status client


To use "IDD_MLP_Tensorflow/jobs/IDD_MLP_Tensorflow" you will need to copy the data folder "my_data_result" from the folder "Flower/ intrusion_detection_dataset_flower" into the folder "NvFlare/My_Nvflare_Custom_Works_localhost_PocMode/poc/admin/transfer/IDD_MLP_Tensorflow/jobs/IDD_MLP_Tensorflow/"