dname=$1

case $dname in
    "nerf_synthetic")
        gdown https://drive.google.com/uc?id=18JxhpWD-4ZmuFKLzKlAw-w5PpzZxXOcG
        unzip nerf_synthetic.zip
        rm -rf __MACOSX
        mv nerf_synthetic data/blender
        rm nerf_synthetic.zip
        ;;
    "nerf_llff")
        gdown https://drive.google.com/uc?id=16VnMcF1KJYxN9QId6TClMsZRahHNMW5g
        unzip nerf_llff_data.zip
        rm -rf __MACOSX
        mv nerf_llff_data data/llff
        rm nerf_llff_data.zip
        ;;
    "nerf_real_360")
        gdown https://drive.google.com/uc?id=1jzggQ7IPaJJTKx9yLASWHrX8dXHnG5eB
        unzip nerf_real_360.zip
        rm -rf __MACOSX
        mkdir nerf_real_360
        mv vasedeck nerf_real_360
        mv pinecone nerf_real_360
        mv nerf_real_360 data/nerf_360
        rm nerf_real_360.zip
        ;;
    "tanks_and_temples")
        gdown 11KRfN91W1AxAW6lOFs4EeYDbeoQZCi87
        unzip tanks_and_temples.zip
        cd tanks_and_temples/tat_training_Truck
        cp -r "test" "validation"
        cd ../..
        mv tanks_and_temples data
        rm tanks_and_temples.zip
        rm -rf __MACOSX
        ;;
    "lf")
        gdown 1gsjDjkbTh4GAR9fFqlIDZ__qR9NYTURQ
        unzip lf_data.zip
        mv lf_data data
        rm -rf __MACOSX
        rm lf_data.zip
        ;;
    "nerf_360_v2")
        wget http://storage.googleapis.com/gresearch/refraw360/360_v2.zip
        mkdir 360_v2
        unzip 360_v2.zip -d 360_v2
        mv 360_v2 data
        rm 360_v2.zip
        ;;
    "shiny_blender")
        wget https://storage.googleapis.com/gresearch/refraw360/ref.zip
        unzip ref.zip
        mv refnerf data
        rm ref.zip
        python utils/preprocess_shiny_blender.py
        ;;
esac
