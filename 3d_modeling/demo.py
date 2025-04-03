import pathlib
import pycolmap


output_path = pathlib.Path(r'E:\data\thesis\Drone Photos\Building 3\1\infront_model')
image_dir =  pathlib.Path(r'E:\data\thesis\Drone Photos\Building 3\1\infront')

output_path.mkdir()
mvs_path = output_path / "mvs"
database_path = output_path / "database.db"

pycolmap.extract_features(database_path, image_dir, sift_options={"max_num_features": 512})
pycolmap.match_exhaustive(database_path)
maps = pycolmap.incremental_mapping(database_path, image_dir, output_path)
maps[0].write(output_path)
# dense reconstruction
pycolmap.undistort_images(mvs_path, output_path, image_dir)
pycolmap.patch_match_stereo(mvs_path)  # requires compilation with CUDA
pycolmap.stereo_fusion(mvs_path / "dense.ply", mvs_path)


# help(pycolmap.SiftExtractionOptions)
# pycolmap.run_pipeline(
#     r"E:\data\thesis\Drone Photos\Building 3\1\infront",
#     r"E:\data\thesis\Drone Photos\Building 3\1\infront_model")