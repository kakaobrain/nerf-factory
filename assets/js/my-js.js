base_url = "https://huggingface.co/nrtf/nerf_factory/resolve/main/"

function select_model___(target, dir, dataset, scene) { 
    if (dir == "left"){
        var img = document.getElementById("left_img" + "_" + dataset + "_" + scene);
        var caption = document.getElementById("left_caption" + "_" + dataset + "_" + scene);
    }
    else if (dir == "right"){
        var img = document.getElementById("right_img" + "_" + dataset + "_" + scene);
        var caption = document.getElementById("right_caption" + "_" + dataset + "_" + scene);
    }
    var model = target.value;
    //img.src = "../../../assets/images/visualization/" + dataset + "/" + model  + "_" + scene + ".jpg";
	img.src = base_url + model + "_" + dataset + "_" + scene + "_220901/render_model/image000.jpg"

}

function select_model(target, dir, dataset, scene) { 
    if (dir == "left"){
        var img = document.getElementById("left_img" + "_" + dataset + "_" + scene);
        var caption = document.getElementById("left_caption" + "_" + dataset + "_" + scene);
    }
    else if (dir == "right"){
        var img = document.getElementById("right_img" + "_" + dataset + "_" + scene);
        var caption = document.getElementById("right_caption" + "_" + dataset + "_" + scene);
    }
    var model = target.value;
	var frame = document.getElementById("input" + "_" + dataset + "_" + scene).value.toString().padStart(3, '0');

	img.src = base_url + model + "_" + dataset + "_" + scene + "_220901/render_model/image" + frame + ".jpg"
    caption.innerHTML = get_psnr(model, dataset, scene)
}


function select_frame(target, dataset, scene) {
	var left_model = document.getElementById("left_select" + "_" + dataset + "_" + scene).value;
	var right_model = document.getElementById("right_select" + "_" + dataset + "_" + scene).value;

	var left_img = document.getElementById("left_img" + "_" + dataset + "_" + scene);
	var right_img = document.getElementById("right_img" + "_" + dataset + "_" + scene);

	var frame = target.value.toString().padStart(3, '0');

	left_img.src =  base_url + left_model + "_" + dataset + "_" + scene + "_220901/render_model/image" + frame + ".jpg"
	right_img.src =  base_url + right_model + "_" + dataset + "_" + scene + "_220901/render_model/image" + frame + ".jpg"
}

function get_psnr(model, dataset, scene){
	if(model == "nerf"){
		if(dataset == "blender"){
			if(scene == "chair")
				return "<b>NeRF</b><br>PSNR: 34.93"
			else if(scene == "drums")
				return "<b>NeRF</b><br>PSNR: 25.28"
			else if(scene == "ficus")
				return "<b>NeRF</b><br>PSNR: 31.28"
			else if(scene == "hotdog")
				return "<b>NeRF</b><br>PSNR: 37.16"
			else if(scene == "lego")
				return "<b>NeRF</b><br>PSNR: 34.38"
			else if(scene == "materials")
				return "<b>NeRF</b><br>PSNR: 30.45"
			else if(scene == "mic")
				return "<b>NeRF</b><br>PSNR: 35.18"
			else if(scene == "ship")
				return "<b>NeRF</b><br>PSNR: 29.95"
		}
		else if(dataset == "blender_multiscale"){
			if(scene == "chair")
				return "<b>NeRF</b><br>PSNR: 32.83"
			else if(scene == "drums")
				return "<b>NeRF</b><br>PSNR: 25.24"
			else if(scene == "ficus")
				return "<b>NeRF</b><br>PSNR: 30.23"
			else if(scene == "hotdog")
				return "<b>NeRF</b><br>PSNR: 35.24"
			else if(scene == "lego")
				return "<b>NeRF</b><br>PSNR: 31.45"
			else if(scene == "materials")
				return "<b>NeRF</b><br>PSNR: 29.54"
			else if(scene == "mic")
				return "<b>NeRF</b><br>PSNR: 32.20"
			else if(scene == "ship")
				return "<b>NeRF</b><br>PSNR: 29.41"
		}
		else if(dataset == "llff"){
			if(scene == "fern")
				return "<b>NeRF</b><br>PSNR: 25.19"
			else if(scene == "flower")
				return "<b>NeRF</b><br>PSNR: 27.94"
			else if(scene == "fortress")
				return "<b>NeRF</b><br>PSNR: 31.73"
			else if(scene == "horns")
				return "<b>NeRF</b><br>PSNR: 28.03"
			else if(scene == "leaves")
				return "<b>NeRF</b><br>PSNR: 21.17"
			else if(scene == "orchids")
				return "<b>NeRF</b><br>PSNR: 20.29"
			else if(scene == "room")
				return "<b>NeRF</b><br>PSNR: 32.96"
			else if(scene == "trex")
				return "<b>NeRF</b><br>PSNR: 27.52"
		}
		else if(dataset == "tanks_and_temples"){
			if(scene == "tat_intermediate_M60")
				return "<b>NeRF</b><br>PSNR: 18.27"
			else if(scene == "tat_intermediate_Playground")
				return "<b>NeRF</b><br>PSNR: 21.68"
			else if(scene == "tat_intermediate_Train")
				return "<b>NeRF</b><br>PSNR: 17.37"
			else if(scene == "tat_training_Truck")
				return "<b>NeRF</b><br>PSNR: 21.44"
		}
		else if(dataset == "lf"){
			if(scene == "africa")
				return "<b>NeRF</b><br>PSNR: 28.53"
			else if(scene == "basket")
				return "<b>NeRF</b><br>PSNR: 21.64"
			else if(scene == "ship")
				return "<b>NeRF</b><br>PSNR: 26.26"
			else if(scene == "statue")
				return "<b>NeRF</b><br>PSNR: 29.76"
			else if(scene == "torch")
				return "<b>NeRF</b><br>PSNR: 23.24"
		}
		else if(dataset == "nerf_360_v2"){
			if(scene == "bicycle")
				return "<b>NeRF</b><br>PSNR: 21.82"
			else if(scene == "bonsai")
				return "<b>NeRF</b><br>PSNR: 29.03"
			else if(scene == "counter")
				return "<b>NeRF</b><br>PSNR: 26.98"
			else if(scene == "garden")
				return "<b>NeRF</b><br>PSNR: 23.64"
			else if(scene == "kitchen")
				return "<b>NeRF</b><br>PSNR: 27.16"
			else if(scene == "room")
				return "<b>NeRF</b><br>PSNR: 30.10"
			else if(scene == "stump")
				return "<b>NeRF</b><br>PSNR: 22.93"
		}
		else if(dataset == "shiny_blender"){
			if(scene == "ball")
				return "<b>NeRF</b><br>PSNR: 27.18"
			else if(scene == "car")
				return "<b>NeRF</b><br>PSNR: 26.42"
			else if(scene == "coffee")
				return "<b>NeRF</b><br>PSNR: 30.64"
			else if(scene == "helmet")
				return "<b>NeRF</b><br>PSNR: 27.61"
			else if(scene == "teapot")
				return "<b>NeRF</b><br>PSNR: 45.37"
			else if(scene == "toaster")
				return "<b>NeRF</b><br>PSNR: 22.51"
		}
	}
	else if(model == "nerfpp"){
		if(dataset == "tanks_and_temples"){
			if(scene == "tat_intermediate_M60")
				return "<b>NeRF++</b><br>PSNR: 17.49"
			else if(scene == "tat_intermediate_Playground")
				return "<b>NeRF++</b><br>PSNR: 22.51"
			else if(scene == "tat_intermediate_Train")
				return "<b>NeRF++</b><br>PSNR: 17.90"
			else if(scene == "tat_training_Truck")
				return "<b>NeRF++</b><br>PSNR: 21.86"
		}
		else if(dataset == "lf"){
			if(scene == "africa")
				return "<b>NeRF++</b><br>PSNR: 28.29"
			else if(scene == "basket")
				return "<b>NeRF++</b><br>PSNR: 21.49"
			else if(scene == "ship")
				return "<b>NeRF++</b><br>PSNR: 26.17"
			else if(scene == "statue")
				return "<b>NeRF++</b><br>PSNR: 30.14"
			else if(scene == "torch")
				return "<b>NeRF++</b><br>PSNR: 25.04"
		}
		else if(dataset == "nerf_360_v2"){
			if(scene == "bicycle")
				return "<b>NeRF++</b><br>PSNR: 21.43"
			else if(scene == "bonsai")
				return "<b>NeRF++</b><br>PSNR: 31.67"
			else if(scene == "counter")
				return "<b>NeRF++</b><br>PSNR: 27.72"
			else if(scene == "garden")
				return "<b>NeRF++</b><br>PSNR: 24.80"
			else if(scene == "kitchen")
				return "<b>NeRF++</b><br>PSNR: 29.47"
			else if(scene == "room")
				return "<b>NeRF++</b><br>PSNR: 30.62"
			else if(scene == "stump")
				return "<b>NeRF++</b><br>PSNR: 24.77"
		}
	}
	else if(model == "plenoxel"){
		if(dataset == "blender"){
			if(scene == "chair")
				return "<b>Plenoxels</b><br>PSNR: 34.11"
			else if(scene == "drums")
				return "<b>Plenoxels</b><br>PSNR: 25.42"
			else if(scene == "ficus")
				return "<b>Plenoxels</b><br>PSNR: 31.94"
			else if(scene == "hotdog")
				return "<b>Plenoxels</b><br>PSNR: 36.69"
			else if(scene == "lego")
				return "<b>Plenoxels</b><br>PSNR: 34.34"
			else if(scene == "materials")
				return "<b>Plenoxels</b><br>PSNR: 29.26"
			else if(scene == "mic")
				return "<b>Plenoxels</b><br>PSNR: 33.36"
			else if(scene == "ship")
				return "<b>Plenoxels</b><br>PSNR: 30.10"
		}
		else if(dataset == "blender_multiscale"){
			if(scene == "chair")
				return "<b>Plenoxels</b><br>PSNR: 32.87"
			else if(scene == "drums")
				return "<b>Plenoxels</b><br>PSNR: 25.34"
			else if(scene == "ficus")
				return "<b>Plenoxels</b><br>PSNR: 30.35"
			else if(scene == "hotdog")
				return "<b>Plenoxels</b><br>PSNR: 34.89"
			else if(scene == "lego")
				return "<b>Plenoxels</b><br>PSNR: 31.42"
			else if(scene == "materials")
				return "<b>Plenoxels</b><br>PSNR: 28.48"
			else if(scene == "mic")
				return "<b>Plenoxels</b><br>PSNR: 31.59"
			else if(scene == "ship")
				return "<b>Plenoxels</b><br>PSNR: 29.04"
		}
		else if(dataset == "llff"){
			if(scene == "fern")
				return "<b>Plenoxels</b><br>PSNR: 24.91"
			else if(scene == "flower")
				return "<b>Plenoxels</b><br>PSNR: 28.17"
			else if(scene == "fortress")
				return "<b>Plenoxels</b><br>PSNR: 31.36"
			else if(scene == "horns")
				return "<b>Plenoxels</b><br>PSNR: 27.77"
			else if(scene == "leaves")
				return "<b>Plenoxels</b><br>PSNR: 21.19"
			else if(scene == "orchids")
				return "<b>Plenoxels</b><br>PSNR: 20.01"
			else if(scene == "room")
				return "<b>Plenoxels</b><br>PSNR: 31.33"
			else if(scene == "trex")
				return "<b>Plenoxels</b><br>PSNR: 26.54"
		}
		else if(dataset == "tanks_and_temples"){
			if(scene == "tat_intermediate_M60")
				return "<b>Plenoxels</b><br>PSNR: 17.74"
			else if(scene == "tat_intermediate_Playground")
				return "<b>Plenoxels</b><br>PSNR: 22.62"
			else if(scene == "tat_intermediate_Train")
				return "<b>Plenoxels</b><br>PSNR: 17.66"
			else if(scene == "tat_training_Truck")
				return "<b>Plenoxels</b><br>PSNR: 22.52"
		}
		else if(dataset == "lf"){
			if(scene == "africa")
				return "<b>Plenoxels</b><br>PSNR: 14.99"
			else if(scene == "basket")
				return "<b>Plenoxels</b><br>PSNR: 16.02"
			else if(scene == "ship")
				return "<b>Plenoxels</b><br>PSNR: 25.40"
			else if(scene == "statue")
				return "<b>Plenoxels</b><br>PSNR: 28.96"
			else if(scene == "torch")
				return "<b>Plenoxels</b><br>PSNR: 24.84"
		}
		else if(dataset == "nerf_360_v2"){
			if(scene == "bicycle")
				return "<b>Plenoxels</b><br>PSNR: 21.42"
			else if(scene == "bonsai")
				return "<b>Plenoxels</b><br>PSNR: 26.21"
			else if(scene == "counter")
				return "<b>Plenoxels</b><br>PSNR: 25.62"
			else if(scene == "garden")
				return "<b>Plenoxels</b><br>PSNR: 23.13"
			else if(scene == "kitchen")
				return "<b>Plenoxels</b><br>PSNR: 25.09"
			else if(scene == "room")
				return "<b>Plenoxels</b><br>PSNR: 28.35"
			else if(scene == "stump")
				return "<b>Plenoxels</b><br>PSNR: 22.88"
		}
		else if(dataset == "shiny_blender"){
			if(scene == "ball")
				return "<b>Plenoxels</b><br>PSNR: 24.52"
			else if(scene == "car")
				return "<b>Plenoxels</b><br>PSNR: 26.11"
			else if(scene == "coffee")
				return "<b>Plenoxels</b><br>PSNR: 31.55"
			else if(scene == "helmet")
				return "<b>Plenoxels</b><br>PSNR: 26.94"
			else if(scene == "teapot")
				return "<b>Plenoxels</b><br>PSNR: 44.25"
			else if(scene == "toaster")
				return "<b>Plenoxels</b><br>PSNR: 19.50"
		}
	}
	else if(model == "dvgo"){
		if(dataset == "blender"){
			if(scene == "chair")
				return "<b>DVGO</b><br>PSNR: 32.69"
			else if(scene == "drums")
				return "<b>DVGO</b><br>PSNR: 25.26"
			else if(scene == "ficus")
				return "<b>DVGO</b><br>PSNR: 32.05"
			else if(scene == "hotdog")
				return "<b>DVGO</b><br>PSNR: 36.11"
			else if(scene == "lego")
				return "<b>DVGO</b><br>PSNR: 33.45"
			else if(scene == "materials")
				return "<b>DVGO</b><br>PSNR: 29.14"
			else if(scene == "mic")
				return "<b>DVGO</b><br>PSNR: 31.99"
			else if(scene == "ship")
				return "<b>DVGO</b><br>PSNR: 28.55"
		}
		else if(dataset == "llff"){
			if(scene == "fern")
				return "<b>DVGO</b><br>PSNR: 25.32"
			else if(scene == "flower")
				return "<b>DVGO</b><br>PSNR: 28.21"
			else if(scene == "fortress")
				return "<b>DVGO</b><br>PSNR: 30.62"
			else if(scene == "horns")
				return "<b>DVGO</b><br>PSNR: 27.73"
			else if(scene == "leaves")
				return "<b>DVGO</b><br>PSNR: 21.45"
			else if(scene == "orchids")
				return "<b>DVGO</b><br>PSNR: 20.66"
			else if(scene == "room")
				return "<b>DVGO</b><br>PSNR: 31.68"
			else if(scene == "trex")
				return "<b>DVGO</b><br>PSNR: 27.16"
		}
		else if(dataset == "tanks_and_temples"){
			if(scene == "tat_intermediate_M60")
				return "<b>DVGO</b><br>PSNR: 17.29"
			else if(scene == "tat_intermediate_Playground")
				return "<b>DVGO</b><br>PSNR: 22.62"
			else if(scene == "tat_intermediate_Train")
				return "<b>DVGO</b><br>PSNR: 17.78"
			else if(scene == "tat_training_Truck")
				return "<b>DVGO</b><br>PSNR: 21.31"
		}
		else if(dataset == "lf"){
			if(scene == "africa")
				return "<b>DVGO</b><br>PSNR: 27.76"
			else if(scene == "basket")
				return "<b>DVGO</b><br>PSNR: 21.28"
			else if(scene == "ship")
				return "<b>DVGO</b><br>PSNR: 25.62"
			else if(scene == "statue")
				return "<b>DVGO</b><br>PSNR: 29.34"
			else if(scene == "torch")
				return "<b>DVGO</b><br>PSNR: 24.15"
		}
		else if(dataset == "nerf_360_v2"){
			if(scene == "bicycle")
				return "<b>DVGO</b><br>PSNR: 21.65"
			else if(scene == "bonsai")
				return "<b>DVGO</b><br>PSNR: 27.92"
			else if(scene == "counter")
				return "<b>DVGO</b><br>PSNR: 26.43"
			else if(scene == "garden")
				return "<b>DVGO</b><br>PSNR: 23.85"
			else if(scene == "kitchen")
				return "<b>DVGO</b><br>PSNR: 26.28"
			else if(scene == "room")
				return "<b>DVGO</b><br>PSNR: 29.11"
			else if(scene == "stump")
				return "<b>DVGO</b><br>PSNR: 20.99"
		}
		else if(dataset == "shiny_blender"){
			if(scene == "ball")
				return "<b>DVGO</b><br>PSNR: 26.13"
			else if(scene == "car")
				return "<b>DVGO</b><br>PSNR: 26.90"
			else if(scene == "coffee")
				return "<b>DVGO</b><br>PSNR: 31.48"
			else if(scene == "helmet")
				return "<b>DVGO</b><br>PSNR: 27.75"
			else if(scene == "teapot")
				return "<b>DVGO</b><br>PSNR: 44.79"
			else if(scene == "toaster")
				return "<b>DVGO</b><br>PSNR: 22.18"
		}
	}
	else if(model == "mipnerf"){
		if(dataset == "blender"){
			if(scene == "chair")
				return "<b>Mip-NeRF</b><br>PSNR: 35.20"
			else if(scene == "drums")
				return "<b>Mip-NeRF</b><br>PSNR: 25.53"
			else if(scene == "ficus")
				return "<b>Mip-NeRF</b><br>PSNR: 33.23"
			else if(scene == "hotdog")
				return "<b>Mip-NeRF</b><br>PSNR: 37.44"
			else if(scene == "lego")
				return "<b>Mip-NeRF</b><br>PSNR: 35.80"
			else if(scene == "materials")
				return "<b>Mip-NeRF</b><br>PSNR: 30.58"
			else if(scene == "mic")
				return "<b>Mip-NeRF</b><br>PSNR: 36.41"
			else if(scene == "ship")
				return "<b>Mip-NeRF</b><br>PSNR: 30.52"
		}
		else if(dataset == "blender_multiscale"){
			if(scene == "chair")
				return "<b>Mip-NeRF</b><br>PSNR: 37.36"
			else if(scene == "drums")
				return "<b>Mip-NeRF</b><br>PSNR: 27.12"
			else if(scene == "ficus")
				return "<b>Mip-NeRF</b><br>PSNR: 33.00"
			else if(scene == "hotdog")
				return "<b>Mip-NeRF</b><br>PSNR: 39.36"
			else if(scene == "lego")
				return "<b>Mip-NeRF</b><br>PSNR: 35.71"
			else if(scene == "materials")
				return "<b>Mip-NeRF</b><br>PSNR: 32.63"
			else if(scene == "mic")
				return "<b>Mip-NeRF</b><br>PSNR: 37.93"
			else if(scene == "ship")
				return "<b>Mip-NeRF</b><br>PSNR: 33.24"
		}
		else if(dataset == "llff"){
			if(scene == "fern")
				return "<b>Mip-NeRF</b><br>PSNR: 24.92"
			else if(scene == "flower")
				return "<b>Mip-NeRF</b><br>PSNR: 27.80"
			else if(scene == "fortress")
				return "<b>Mip-NeRF</b><br>PSNR: 31.73"
			else if(scene == "horns")
				return "<b>Mip-NeRF</b><br>PSNR: 27.79"
			else if(scene == "leaves")
				return "<b>Mip-NeRF</b><br>PSNR: 20.94"
			else if(scene == "orchids")
				return "<b>Mip-NeRF</b><br>PSNR: 20.27"
			else if(scene == "room")
				return "<b>Mip-NeRF</b><br>PSNR: 33.24"
			else if(scene == "trex")
				return "<b>Mip-NeRF</b><br>PSNR: 27.69"
		}
		else if(dataset == "tanks_and_temples"){
			if(scene == "tat_intermediate_M60")
				return "<b>Mip-NeRF</b><br>PSNR: 18.41"
			else if(scene == "tat_intermediate_Playground")
				return "<b>Mip-NeRF</b><br>PSNR: 21.83"
			else if(scene == "tat_intermediate_Train")
				return "<b>Mip-NeRF</b><br>PSNR: 17.87"
			else if(scene == "tat_training_Truck")
				return "<b>Mip-NeRF</b><br>PSNR: 21.71"
		}
		else if(dataset == "lf"){
			if(scene == "africa")
				return "<b>Mip-NeRF</b><br>PSNR: 28.65"
			else if(scene == "basket")
				return "<b>Mip-NeRF</b><br>PSNR: 21.98"
			else if(scene == "ship")
				return "<b>Mip-NeRF</b><br>PSNR: 26.43"
			else if(scene == "statue")
				return "<b>Mip-NeRF</b><br>PSNR: 29.86"
			else if(scene == "torch")
				return "<b>Mip-NeRF</b><br>PSNR: 23.29"
		}
		else if(dataset == "shiny_blender"){
			if(scene == "ball")
				return "<b>Mip-NeRF</b><br>PSNR: 27.29"
			else if(scene == "car")
				return "<b>Mip-NeRF</b><br>PSNR: 26.72"
			else if(scene == "coffee")
				return "<b>Mip-NeRF</b><br>PSNR: 30.83"
			else if(scene == "helmet")
				return "<b>Mip-NeRF</b><br>PSNR: 27.79"
			else if(scene == "teapot")
				return "<b>Mip-NeRF</b><br>PSNR: 45.50"
			else if(scene == "toaster")
				return "<b>Mip-NeRF</b><br>PSNR: 22.52"
		}
		else if(dataset == "ref_real"){
			if(scene == "gardenspheres")
				return "<b>Mip-NeRF</b><br>PSNR: 16.44"
			else if(scene == "sedan")
				return "<b>Mip-NeRF</b><br>PSNR: 20.57"
			else if(scene == "toycar")
				return "<b>Mip-NeRF</b><br>PSNR: 15.04"
		}
		else if(dataset == "nerf_360_v2"){
			if(scene == "bicycle")
				return "<b>Mip-NeRF</b><br>PSNR: 21.72"
			else if(scene == "bonsai")
				return "<b>Mip-NeRF</b><br>PSNR: 29.12"
			else if(scene == "counter")
				return "<b>Mip-NeRF</b><br>PSNR: 26.77"
			else if(scene == "garden")
				return "<b>Mip-NeRF</b><br>PSNR: 23.71"
			else if(scene == "kitchen")
				return "<b>Mip-NeRF</b><br>PSNR: 27.98"
			else if(scene == "room")
				return "<b>Mip-NeRF</b><br>PSNR: 30.23"
			else if(scene == "stump")
				return "<b>Mip-NeRF</b><br>PSNR: 22.74"
		}
	}
	else if(model == "refnerf"){
		if(dataset == "blender"){
			if(scene == "chair")
				return "<b>Ref-NeRF</b><br>PSNR: 35.84"
			else if(scene == "drums")
				return "<b>Ref-NeRF</b><br>PSNR: 25.52"
			else if(scene == "ficus")
				return "<b>Ref-NeRF</b><br>PSNR: 31.32"
			else if(scene == "hotdog")
				return "<b>Ref-NeRF</b><br>PSNR: 36.54"
			else if(scene == "lego")
				return "<b>Ref-NeRF</b><br>PSNR: 35.79"
			else if(scene == "materials")
				return "<b>Ref-NeRF</b><br>PSNR: 35.71"
			else if(scene == "mic")
				return "<b>Ref-NeRF</b><br>PSNR: 35.96"
			else if(scene == "ship")
				return "<b>Ref-NeRF</b><br>PSNR: 29.51"
		}
		else if(dataset == "shiny_blender"){
			if(scene == "ball")
				return "<b>Ref-NeRF</b><br>PSNR: 43.09"
			else if(scene == "car")
				return "<b>Ref-NeRF</b><br>PSNR: 30.70"
			else if(scene == "coffee")
				return "<b>Ref-NeRF</b><br>PSNR: 32.27"
			else if(scene == "helmet")
				return "<b>Ref-NeRF</b><br>PSNR: 29.66"
			else if(scene == "teapot")
				return "<b>Ref-NeRF</b><br>PSNR: 45.20"
			else if(scene == "toaster")
				return "<b>Ref-NeRF</b><br>PSNR: 24.88"
		}
	}
	else if(model == "mipnerf360"){
		if(dataset == "llff"){
			if(scene == "fern")
				return "<b>Mip-NeRF 360</b><br>PSNR: 24.58"
			else if(scene == "flower")
				return "<b>Mip-NeRF 360</b><br>PSNR: 27.81"
			else if(scene == "fortress")
				return "<b>Mip-NeRF 360</b><br>PSNR: 31.17"
			else if(scene == "horns")
				return "<b>Mip-NeRF 360</b><br>PSNR: 28.03"
			else if(scene == "leaves")
				return "<b>Mip-NeRF 360</b><br>PSNR: 20.28"
			else if(scene == "orchids")
				return "<b>Mip-NeRF 360</b><br>PSNR: 19.74"
			else if(scene == "room")
				return "<b>Mip-NeRF 360</b><br>PSNR: 33.55"
			else if(scene == "trex")
				return "<b>Mip-NeRF 360</b><br>PSNR: 27.87"
		}
		else if(dataset == "tanks_and_temples"){
			if(scene == "tat_intermediate_M60")
				return "<b>Mip-NeRF 360</b><br>PSNR: 20.09"
			else if(scene == "tat_intermediate_Playground")
				return "<b>Mip-NeRF 360</b><br>PSNR: 24.27"
			else if(scene == "tat_intermediate_Train")
				return "<b>Mip-NeRF 360</b><br>PSNR: 19.74"
			else if(scene == "tat_training_Truck")
				return "<b>Mip-NeRF 360</b><br>PSNR: 24.14"
		}
		else if(dataset == "lf"){
			if(scene == "africa")
				return "<b>Mip-NeRF 360</b><br>PSNR: 29.58"
			else if(scene == "basket")
				return "<b>Mip-NeRF 360</b><br>PSNR: 21.19"
			else if(scene == "ship")
				return "<b>Mip-NeRF 360</b><br>PSNR: 30.16"
			else if(scene == "statue")
				return "<b>Mip-NeRF 360</b><br>PSNR: 34.90"
			else if(scene == "torch")
				return "<b>Mip-NeRF 360</b><br>PSNR: 25.86"
		}
		else if(dataset == "nerf_360_v2"){
			if(scene == "bicycle")
				return "<b>Mip-NeRF 360</b><br>PSNR: 22.86"
			else if(scene == "bonsai")
				return "<b>Mip-NeRF 360</b><br>PSNR: 32.97"
			else if(scene == "counter")
				return "<b>Mip-NeRF 360</b><br>PSNR: 29.29"
			else if(scene == "garden")
				return "<b>Mip-NeRF 360</b><br>PSNR: 26.01"
			else if(scene == "kitchen")
				return "<b>Mip-NeRF 360</b><br>PSNR: 31.99"
			else if(scene == "room")
				return "<b>Mip-NeRF 360</b><br>PSNR: 32.68"
			else if(scene == "stump")
				return "<b>Mip-NeRF 360</b><br>PSNR: 25.28"
		}
	}
}
