"""
Demo render.
1. save / load textured .obj file
2. render using SoftRas with different sigma / gamma
"""
import matplotlib.pyplot as plt
import os
import tqdm
import numpy as np
import imageio
import argparse

import soft_renderer as sr


current_dir = os.path.dirname(os.path.realpath(__file__))
data_dir = os.path.join(current_dir, '../data')


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('-i', '--filename-input', type=str,
                        default=os.path.join(data_dir, 'obj/spot/spot_triangulated.obj'))
    parser.add_argument('-o', '--output-dir', type=str,
                        default=os.path.join(data_dir, 'results/output_render'))
    args = parser.parse_args()

    # other settings
    camera_distance = 2.732
    elevation = 30
    azimuth = 0

    # load from Wavefront .obj file
    mesh_ = sr.Mesh.from_obj(args.filename_input,
                             load_texture=True, texture_res=5, texture_type='surface')

    # create renderer with SoftRas
    transform = sr.LookAt()
    lighting = sr.Lighting()
    rasterizer = sr.SoftRasterizer()

    os.makedirs(args.output_dir, exist_ok=True)

    # draw object from different view
    loop = tqdm.tqdm(list(range(0, 360, 4)))
    writer = imageio.get_writer(os.path.join(args.output_dir, 'rotation.gif'), mode='I')
    for num, azimuth in enumerate(loop):
        loop.set_description('Drawing rotation')

        # render mesh
        mesh = lighting(mesh_)
        transform.set_eyes_from_angles(camera_distance, elevation, azimuth)
        mesh = transform(mesh)
        images = rasterizer(mesh)

        image = images.detach().cpu().numpy()[0].transpose((1, 2, 0))
        writer.append_data((255*image).astype(np.uint8))
    writer.close()

    # draw object from different sigma and gamma
    loop = tqdm.tqdm(list(np.arange(-4, -2, 0.2)))
    transform.set_eyes_from_angles(camera_distance, elevation, 45)
    writer = imageio.get_writer(os.path.join(args.output_dir, 'bluring.gif'), mode='I')
    for num, gamma_pow in enumerate(loop):

        # render mesh
        mesh = lighting(mesh_)
        mesh = transform(mesh)
        rasterizer.set_gamma(10**gamma_pow)
        rasterizer.set_sigma(10**(gamma_pow - 1))
        loop.set_description('Drawing blurring')
        images = rasterizer(mesh)

        image = images.detach().cpu().numpy()[0].transpose((1, 2, 0))
        writer.append_data((255*image).astype(np.uint8))
    writer.close()

    # save to textured obj
    mesh.save_obj(os.path.join(args.output_dir, 'saved_spot.obj'), save_texture=True)


if __name__ == '__main__':
    main()
