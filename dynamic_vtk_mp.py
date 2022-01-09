#!/usr/bin/env python
import numpy as np
from numpy import random
import numpy.linalg as npl
import threading
import SimpleITK as sitk
import vtk
import time
from vtk.util.numpy_support import *
from vtk.util.vtkImageImportFromArray import *
from dectmkr import *
from p2dto3d import *
from show3d import *

class VtkPointCloud:
    def __init__(self, zMin=-10.0, zMax=10.0, maxNumPoints=1e6):
        self.maxNumPoints = maxNumPoints
        self.vtkPolyData = vtk.vtkPolyData()
        self.clearPoints()
        mapper = vtk.vtkPolyDataMapper()
        mapper.SetInputData(self.vtkPolyData)
        mapper.SetColorModeToDefault()
        mapper.SetScalarRange(zMin, zMax)
        mapper.SetScalarVisibility(1)
        self.vtkActor = vtk.vtkActor()
        self.vtkActor.SetMapper(mapper)
        self.vtkActor.GetProperty().SetPointSize(3)

    def addPoint(self, point):
        if self.vtkPoints.GetNumberOfPoints() < self.maxNumPoints:
            pointId = self.vtkPoints.InsertNextPoint(point[:])
            self.vtkDepth.InsertNextValue(point[2])
            self.vtkCells.InsertNextCell(1)
            self.vtkCells.InsertCellPoint(pointId)
        else:
            r = random.randint(0, self.maxNumPoints)
            self.vtkPoints.SetPoint(r, point[:])

        self.vtkCells.Modified()
        self.vtkPoints.Modified()
        self.vtkDepth.Modified()

    def clearPoints(self):
        self.vtkPoints = vtk.vtkPoints()
        self.vtkCells = vtk.vtkCellArray()
        self.vtkDepth = vtk.vtkDoubleArray()
        self.vtkDepth.SetName('DepthArray')

        self.vtkPolyData.SetPoints(self.vtkPoints)
        self.vtkPolyData.SetVerts(self.vtkCells)
        self.vtkPolyData.GetPointData().SetScalars(self.vtkDepth)
        self.vtkPolyData.GetPointData().SetActiveScalars('DepthArray')


wire_data = {
    'points': []
}

# class WireThread(threading.Thread):
#     def __init__(self, d):
#         super().__init__()
#         self.d = d

#     def run(self):
#         while True:
#             points = []
#             for k in range(100):
#                 point = 20*(random.rand(3)-0.5)
#                 points.append(point)

#             self.d['points'] = points
#             print (points)
#             time.sleep(0.1)

class WireThread(threading.Thread):
    def __init__(self, d):
        super().__init__()
        self.d = d
        path_3dmrk_json = 'data_0108/Static 3D_0108/F_1.mrk.json'
        path_2dmrk_nii = 'data_0108/Static 2D_0108/newmarkers_2d.nii.gz'
        p3d_list = get3dp(path_3dmrk_json)
        p2d_np= get2dp(path_2dmrk_nii)
        self.points = np.array(p3d_list)
        p3d_np = np.array(p3d_list).T
        # p3d_np = np.flip(p3d_np, axis=1)
        self.curve3d = gen_curve(p3d_np)
        self.curve2d = gen_curve(p2d_np)

        f_path = 'data_0108/Dynamic 2D_0108/2DMoving_1'
        save_path = '0108_2dmoving_bgsub'
        os.makedirs(save_path, exist_ok=True) 
        self.im_list = read_img(f_path)
        self.bg_mask = get_bg_mask(self.im_list) 

    def run(self):
        im_list = self.im_list
        mask = self.bg_mask
        curve2d = self.curve2d
        curve3d = self.curve3d

        a = im_list[1].astype(np.int32)
        cx, cy, count = process_one(a, mask)
        pre_p = [cx, cy]
        # print (pre_p)
        # raise
        xyz = []
        i = 1
        while True:
        # for i in range(2, len(im_list)):
            i = i+1 if i < len(im_list) else i
            a = im_list[i].astype(np.int32)
            cx, cy, count = process_one(a, mask, pre_p, count)
            pre_p = [cx, cy]
            t = get_t(pre_p, curve2d)
            # print (t)
            res_3dp = curve3d.evaluate(t)
            # print (res_3dp)
            res_3dp = np.squeeze(res_3dp)
            # res_3dp = np.array([res_3dp[1], res_3dp[0],res_3dp[2]])
            xyz.append(res_3dp)
            # xyz = [np.squeeze(res_3dp)]
            print (xyz)
            # xyz_np = np.array(xyz)  
            self.d['points'] = xyz #self.points #xyz
            # time.sleep(0.1)
        


class AddPointCloudTimerCallback():
    def __init__(self, renderer, iterations, d):
        self.iterations = iterations
        self.renderer = renderer
        self.prev_actor = None
        self.d = d

    def execute(self, iren, event):
        if self.iterations == 0:
            iren.DestroyTimer(self.timerId)

        pc = VtkPointCloud()
        if self.prev_actor:
            self.renderer.RemoveActor(self.prev_actor)
        self.renderer.AddActor(pc.vtkActor)
        self.prev_actor = pc.vtkActor

        pc.clearPoints()

        pts = self.d['points']
        for i in pts:
            pc.addPoint(i)

        iren.GetRenderWindow().Render()

        self.iterations -= 1

def load_aota(filename):
    reader = vtk.vtkXMLPolyDataReader()
    reader.SetFileName(filename)
    reader.Update()

    pd = reader.GetOutput()

    scalars = pd.GetPointData().GetScalars("ImageScalars")
    np_scalars = vtk_to_numpy(scalars)
    scalars = numpy_to_vtk(np.array([255]*len(np_scalars), dtype='u1'))

    pd.GetPointData().SetScalars(scalars)
    return pd

if __name__ == "__main__":
    # path = 'static_3d_crossS.nii.gz'

    aota_f = 'data_0108/aota.vtp'
    pd_aota = load_aota(aota_f)

    r = vtk.vtkRenderer()
    r.SetBackground(0.5, 0.6, 0.8)

    mapper = vtk.vtkPolyDataMapper()
    mapper.SetInputData(pd_aota)

    actor = vtk.vtkActor()
    actor.SetMapper(mapper)
    actor.GetProperty().SetOpacity(0.3)
    r.AddActor(actor)

    actor = vtk.vtkActor()
    actor.SetMapper(mapper)

    rw = vtk.vtkRenderWindow()
    rw.SetSize(2000, 2000)

    rw.AddRenderer(r)

    rwi = vtk.vtkRenderWindowInteractor()
    rwi.SetRenderWindow(rw)
    rwi.SetInteractorStyle(vtk.vtkInteractorStyleTrackballCamera())
    rwi.Initialize()

    thr = WireThread(wire_data)

    cb = AddPointCloudTimerCallback(r, 3000, wire_data)
    rwi.AddObserver('TimerEvent', cb.execute)
    cb.timerId = rwi.CreateRepeatingTimer(0)

    camera = vtk.vtkCamera()
    camera.SetPosition((500, 500, 500))
    camera.SetViewUp((0, 0, 1))
    camera.SetFocalPoint((0, 0, 1))
    r.SetActiveCamera(camera)


    thr.start()

    rw.Render()
    rwi.Start()
