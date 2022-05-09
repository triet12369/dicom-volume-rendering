import os, sys
import numpy as np
import vtk
from input import *


#!/usr/bin/env python

# noinspection PyUnresolvedReferences
import vtkmodules.vtkInteractionStyle
# noinspection PyUnresolvedReferences
import vtkmodules.vtkRenderingOpenGL2
# noinspection PyUnresolvedReferences
import vtkmodules.vtkRenderingVolumeOpenGL2
from vtkmodules.vtkCommonColor import vtkNamedColors
from vtkmodules.vtkCommonDataModel import vtkPiecewiseFunction
from vtkmodules.vtkIOImage import vtkMetaImageReader
from vtkmodules.vtkIOImage import (
    vtkDICOMImageReader,
)
from vtkmodules.vtkRenderingCore import (
    vtkColorTransferFunction,
    vtkRenderWindow,
    vtkRenderWindowInteractor,
    vtkRenderer,
    vtkVolume,
    vtkVolumeProperty,
)
from vtkmodules.vtkRenderingVolume import vtkGPUVolumeRayCastMapper

from input import *
from src.model.colormap.Standard import STANDARD
from src.model.colormap.toRGBPoints import to_rgb_points
from src.export.nerf import export_to_nerf

def main():
    colors = vtkNamedColors()

    args = get_program_parameters()
    file_name = args.dicom_folder

    colors.SetColor('BkgColor', [255, 255, 255, 0])

    # Create the renderer, the render window, and the interactor. The renderer
    # draws into the render window, the interactor enables mouse- and
    # keyboard-based interaction with the scene.
    ren = vtkRenderer()
    ren_win = vtkRenderWindow()
    ren_win.AddRenderer(ren)


    # The following reader is used to read a series of 2D slices (images)
    # that compose the volume. The slice dimensions are set, and the
    # pixel spacing. The data Endianness must also be specified. The reader
    # uses the FilePrefix in combination with the slice number to construct
    # filenames using the format FilePrefix.%d. (In this case the FilePrefix
    # is the root name of the file: quarter.)
    # reader = vtkMetaImageReader()
    # reader.SetFileName(file_name)
    reader = vtkDICOMImageReader()
    reader.SetDirectoryName(file_name)

    # The volume will be displayed by ray-cast alpha compositing.
    # A ray-cast mapper is needed to do the ray-casting.
    volume_mapper = vtkGPUVolumeRayCastMapper()
    volume_mapper.SetInputConnection(reader.GetOutputPort())

    # The color transfer function maps voxel intensities to colors.
    # It is modality-specific, and often anatomy-specific as well.
    # The goal is to one color for flesh (between 500 and 1000)
    # and another color for bone (1150 and over).
    rgb_points = to_rgb_points(STANDARD)
    volume_color = vtkColorTransferFunction()
    # volume_color.AddRGBPoint(0, 0.0, 0.0, 0.0)
    # volume_color.AddRGBPoint(500, 240.0 / 255.0, 184.0 / 255.0, 160.0 / 255.0)
    # volume_color.AddRGBPoint(1000, 240.0 / 255.0, 184.0 / 255.0, 160.0 / 255.0)
    # volume_color.AddRGBPoint(1150, 1.0, 1.0, 240.0 / 255.0)  # Ivory
    for rgb_point in rgb_points:
        volume_color.AddRGBPoint(rgb_point[0], rgb_point[1], rgb_point[2], rgb_point[3])

    # The opacity transfer function is used to control the opacity
    # of different tissue types.
    volume_scalar_opacity = vtkPiecewiseFunction()
    volume_scalar_opacity.AddPoint(0, 0.00)
    volume_scalar_opacity.AddPoint(500, 0.15)
    volume_scalar_opacity.AddPoint(800, 1.00)
    # volume_scalar_opacity.AddPoint(1150, 1.00)

    # The gradient opacity function is used to decrease the opacity
    # in the 'flat' regions of the volume while maintaining the opacity
    # at the boundaries between tissue types.  The gradient is measured
    # as the amount by which the intensity changes over unit distance.
    # For most medical data, the unit distance is 1mm.
    volume_gradient_opacity = vtkPiecewiseFunction()
    volume_gradient_opacity.AddPoint(0, 0.0)
    volume_gradient_opacity.AddPoint(90, 0.5)
    volume_gradient_opacity.AddPoint(100, 1.0)

    # The VolumeProperty attaches the color and opacity functions to the
    # volume, and sets other volume properties.  The interpolation should
    # be set to linear to do a high-quality rendering.  The ShadeOn option
    # turns on directional lighting, which will usually enhance the
    # appearance of the volume and make it look more '3D'.  However,
    # the quality of the shading depends on how accurately the gradient
    # of the volume can be calculated, and for noisy data the gradient
    # estimation will be very poor.  The impact of the shading can be
    # decreased by increasing the Ambient coefficient while decreasing
    # the Diffuse and Specular coefficient.  To increase the impact
    # of shading, decrease the Ambient and increase the Diffuse and Specular.
    volume_property = vtkVolumeProperty()
    volume_property.SetColor(volume_color)
    volume_property.SetScalarOpacity(volume_scalar_opacity)
    volume_property.SetGradientOpacity(volume_gradient_opacity)
    volume_property.SetInterpolationTypeToLinear()
    volume_property.ShadeOn()
    volume_property.SetAmbient(0.4)
    volume_property.SetDiffuse(1.0)
    volume_property.SetSpecular(0.4)

    # Extra paraneeters for volume mapper
    volume_mapper.SetBlendModeToComposite()
    volume_mapper.SetSampleDistance(0.5)
    volume_mapper.AutoAdjustSampleDistancesOff()
    volume_mapper.SetUseJittering(True)
    volume_mapper.UseJitteringOn()

    # The vtkVolume is a vtkProp3D (like a vtkActor) and controls the position
    # and orientation of the volume in world coordinates.
    volume = vtkVolume()
    volume.SetMapper(volume_mapper)
    volume.SetProperty(volume_property)

    # Finally, add the volume to the renderer
    ren.AddViewProp(volume)

    # Set up an initial view of the volume.  The focal point will be the
    # center of the volume, and the camera position will be 400mm to the
    # patient's left (which is our right).
    view_angle = 40.0
    camera = ren.GetActiveCamera()
    c = volume.GetCenter()
    camera.SetViewUp(0, 0, -1)
    camera.SetFocalPoint(c[0], c[1], c[2])
    camera.SetViewAngle(view_angle)

    # Position camera so that the volume fit the camera FOV
    # angle = 2*atan((h/2)/d)
    # d = (h/2)/tan(angle/2)
    # vtk's camera Y-axis is the axis that points towards the scene
    pixel_spacing = reader.GetPixelSpacing()
    max_x = (volume.GetMaxXBound() + 1)
    max_y = (volume.GetMaxYBound() + 1)
    max_z = (volume.GetMaxZBound() + 1)
    max_dim = np.max([max_x, max_y, max_z])
    offset = (max_z / 2) / np.tan(np.radians(view_angle / 2)) + (max_x / 2)
    camera.SetPosition(c[0], c[1] - offset, c[2])
    camera.SetClippingRange(0.1, offset + max_dim)
    # camera.SetClippingRange(2.0, 6.0)
    print("pixel spacing", pixel_spacing)
    print("volume", max_dim, offset)

    # camera.Azimuth(30.0)
    # camera.Elevation(30.0)

    # Set a background color for the renderer
    ren.SetBackground(colors.GetColor3d('BkgColor'))

    # Increase the size of the render window
    ren_win.SetSize(800, 800)
    ren_win.SetWindowName('MedicalDemo4')
    ren_win.SetAlphaBitPlanes(1)

    if args.export_nerf:
        # export_to_folder(None, render_window=ren_win)
        export_to_nerf(camera, render_window=ren_win, show_preview=False)
    else:
        # Interact with the data.
        iren = vtkRenderWindowInteractor()
        iren.SetRenderWindow(ren_win)
        iren.Start()


def get_program_parameters():
    import argparse
    description = 'Read a volume dataset and displays it via volume rendering.'
    epilogue = '''
    Derived from VTK/Examples/Cxx/Medical4.cxx
    '''
    parser = argparse.ArgumentParser(description=description, epilog=epilogue,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--dicom-folder')
    parser.add_argument('--export-nerf', action=argparse.BooleanOptionalAction)
    args = parser.parse_args()
    print("args", args)
    return args


if __name__ == '__main__':
    main()


