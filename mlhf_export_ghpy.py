""" mlhf data converter (from gh_cpython to Rhino Geometry)
    Inputs:
        nsym: int / num of symmetry
        lcount: int / num of layers
        u_step: int / num of u_steps
        r_pp: float tree / r coordinate of point P
        z_pp: float tree / z coordinate of point P
        beta_pp: float tree / arg angle beta of hinge vecctor of point P
        r_qq: float tree / r coordinate of point Q
        z_qq: float tree / z coordinate of point Q
        beta_qq: float tree / arg angle beta of hinge vecctor of point Q
        r_pp_off: float tree / r coordinate of hinge offset point of P
        z_pp_off: float tree / z coordinate of hinge offset point of P
        offangle_pp: float tree / arg angle of hinge offset vector from P
        r_qq_off: float tree / r coordinate of hinge offset point of Q
        z_qq_off: float tree / z coordinate of hinge offset point of Q
        offangle_qq: float tree / arg angle of hinge offset vector from Q
        elem_x_vec_0: float tree / direction component of element coordinate x 
        elem_x_vec_1: float tree / direction component of element coordinate x
        elem_x_vec_2: float tree / direction component of element coordinate x
        elem_y_vec_0: float tree / direction component of element coordinate y 
        elem_y_vec_1: float tree / direction component of element coordinate y
        elem_y_vec_2: float tree / direction component of element coordinate y
        elem_z_vec_0: float tree / direction component of element coordinate z
        elem_z_vec_1: float tree / direction component of element coordinate z
        elem_z_vec_2: float tree / direction component of element coordinate z
        idx_u: int / index of u_list to visualize
    Output:
        vecs_elem_x: vector of element coordinate x
        vecs_elem_y: vector of element coordinate y
        vecs_elem_z: vector of element coordinate z
        pts_pp_all: points of all P
        pts_pp_all: points of all Q
        lines_pq_all: lines between offset points of P and Q
        vecs_hinge_pp: vectors of hinge axis on P
        vecs_hinge_qq: vectors of hinge axis on Q
        vecs_off_pp: vectors of hinge offset on P
        vecs_off_qq: vectors of hinge offset on Q
        """

__author__ = "RyoWATADA"
__version__ = "2022.02.05"

import rhinoscriptsyntax as rs
import ghpythonlib.treehelpers as th
import Rhino.Geometry as rg
import math

import System
from Grasshopper.Kernel.Data import GH_Path
from Grasshopper import DataTree

# GH_CPythonで計算された結果をリストに変換し、RhinoGeometryに変換する

# lcount : 層数
# u_step : パラメータu(0 to 1)の分割成分数
# r_pp_l : 各層の点P（下部節点）のR座標
# z_pp_l : 各層の点P（下部節点）のZ座標
# ...
# idx_u : 出力するパラメータuの状態（0からdiv_u+1までのうちのインデックスで指定）

def cvt_list(_tree, _lc, _u_step):
    _l = [[0 for iu in range(_u_step)] for ic in range(_lc)]
    for ic in range(_lc):
        _l[ic] = th.tree_to_list(_tree, lambda x:x[ic])
    return _l

def pick_state_u(_list, _lc, _idx_u):
    _pick = [0 for ic in range(_lc)]
    for ic in range(_lc):
        _pick[ic] = _list[ic][_idx_u]
    return _pick

def vec_transform(_vecx1, _vecy1, _vecz1, _xform):
    # _xform に従い、3つのベクトルをトランスフォームしたものを返す
    _vecx2 = rs.VectorTransform(_vecx1, _xform)
    _vecy2 = rs.VectorTransform(_vecy1, _xform)
    _vecz2 = rs.VectorTransform(_vecz1, _xform)
    return _vecx2, _vecy2, _vecz2

# Treeをlistに変換
r_pp_l = cvt_list(r_pp, lcount, u_step)
z_pp_l = cvt_list(z_pp, lcount, u_step)
beta_pp_l = cvt_list(beta_pp, lcount, u_step)
r_qq_l = cvt_list(r_qq, lcount, u_step)
z_qq_l = cvt_list(z_qq, lcount, u_step)
beta_qq_l = cvt_list(beta_qq, lcount, u_step)
r_pp_off_l = cvt_list(r_pp_off, lcount, u_step)
z_pp_off_l = cvt_list(z_pp_off, lcount, u_step)
offangle_pp_l = cvt_list(offangle_pp, lcount, u_step)
r_qq_off_l = cvt_list(r_qq_off, lcount, u_step)
z_qq_off_l = cvt_list(z_qq_off, lcount, u_step)
offangle_qq_l = cvt_list(offangle_qq, lcount, u_step)
elem_x_vec_0_l = cvt_list(elem_x_vec_0, lcount, u_step)
elem_x_vec_1_l = cvt_list(elem_x_vec_1, lcount, u_step)
elem_x_vec_2_l = cvt_list(elem_x_vec_2, lcount, u_step)
elem_y_vec_0_l = cvt_list(elem_y_vec_0, lcount, u_step)
elem_y_vec_1_l = cvt_list(elem_y_vec_1, lcount, u_step)
elem_y_vec_2_l = cvt_list(elem_y_vec_2, lcount, u_step)
elem_z_vec_0_l = cvt_list(elem_z_vec_0, lcount, u_step)
elem_z_vec_1_l = cvt_list(elem_z_vec_1, lcount, u_step)
elem_z_vec_2_l = cvt_list(elem_z_vec_2, lcount, u_step)

# RhinoGeometryを作成
pts_pp = [] #各層点Pの座標（RZ平面投影のみ）
tmp_r = pick_state_u(r_pp_l, lcount, idx_u)
tmp_z = pick_state_u(z_pp_l, lcount, idx_u)

for ic in range(lcount):
    pt = rs.AddPoint(tmp_r[ic], 0, tmp_z[ic])
    pts_pp.append(pt)

pts_qq = [] #各層点Qの座標（RZ平面投影のみ）
tmp_r = pick_state_u(r_qq_l, lcount, idx_u)
tmp_z = pick_state_u(z_qq_l, lcount, idx_u)

for ic in range(lcount):
    pt = rs.AddPoint(tmp_r[ic], 0, tmp_z[ic])
    pts_qq.append(pt)

num_beta_pp = pick_state_u(beta_pp_l, lcount, idx_u) #各層点Pの角度beta
num_beta_qq = pick_state_u(beta_qq_l, lcount, idx_u) #各層点Qの角度beta

pts_pp_off = [] #各層点Pオフセットの座標（RZ平面投影のみ）
tmp_r = pick_state_u(r_pp_off_l, lcount, idx_u)
tmp_z = pick_state_u(z_pp_off_l, lcount, idx_u)

for ic in range(lcount):
    pt = rs.AddPoint(tmp_r[ic], 0, tmp_z[ic])
    pts_pp_off.append(pt)

pts_qq_off = [] #各層点Qオフセットの座標（RZ平面投影のみ）
tmp_r = pick_state_u(r_qq_off_l, lcount, idx_u)
tmp_z = pick_state_u(z_qq_off_l, lcount, idx_u)

for ic in range(lcount):
    pt = rs.AddPoint(tmp_r[ic], 0, tmp_z[ic])
    pts_qq_off.append(pt)

num_offangle_pp = pick_state_u(offangle_pp_l, lcount, idx_u) #各層点Pの角度offangle
num_offangle_qq = pick_state_u(offangle_qq_l, lcount, idx_u) #各層点Qの角度offangle


# 要素座標軸ベクトル データツリー作成

# transform定義
# PI/nsym面対称
xform_mirror=rs.XformMirror(rg.Point3d(0,0,0), \
    rg.Vector3d(-math.sin(math.pi/(nsym)),math.cos(math.pi/(nsym)),0))
# 2*PI/nsym回転
xform_rotate=rs.XformRotation2(360/nsym, rg.Vector3d.ZAxis, rg.Point3d(0,0,0))
# PI/nsym回転
xform_rotate_h=rs.XformRotation2(180/nsym, rg.Vector3d.ZAxis, rg.Point3d(0,0,0))

vecs_elem_x = DataTree[System.Object]()
vecs_elem_y = DataTree[System.Object]()
vecs_elem_z = DataTree[System.Object]()

for ic in range(lcount):
    path = GH_Path(0, ic)
    vecx1=rg.Vector3d(elem_x_vec_0_l[ic][idx_u], \
                      elem_x_vec_1_l[ic][idx_u], \
                      elem_x_vec_2_l[ic][idx_u])
    vecy1=rg.Vector3d(elem_y_vec_0_l[ic][idx_u], \
                      elem_y_vec_1_l[ic][idx_u], \
                      elem_y_vec_2_l[ic][idx_u])
    vecz1=rg.Vector3d(elem_z_vec_0_l[ic][idx_u], \
                      elem_z_vec_1_l[ic][idx_u], \
                      elem_z_vec_2_l[ic][idx_u])
    if ic % 2 == 1 : # 奇数層はPI/(nsym*2)面でベクトルを鏡面変換
        xfm=rs.XformMirror(rg.Point3d(0,0,0), \
           rg.Vector3d(-math.sin(math.pi/(2*nsym)),math.cos(math.pi/(2*nsym)), \
           0))
        vecx1, vecy1, vecz1 = vec_transform(vecx1, vecy1, vecz1, xfm)

    vecs_elem_x.Add(vecx1,path)
    vecs_elem_y.Add(vecy1,path)
    vecs_elem_z.Add(vecz1,path)
    vecx2, vecy2, vecz2 = vec_transform(vecx1, vecy1, vecz1, xform_mirror)
    vecs_elem_x.Add(vecx2,path)
    vecs_elem_y.Add(vecy2,path)
    vecs_elem_z.Add(vecz2,path)
    for ip in range(1,nsym):
        vecx1, vecy1, vecz1 = vec_transform(vecx1, vecy1, vecz1, xform_rotate)
        vecx2, vecy2, vecz2 = vec_transform(vecx2, vecy2, vecz2, xform_rotate)
        vecs_elem_x.Add(vecx1,path)
        vecs_elem_x.Add(vecx2,path)
        vecs_elem_y.Add(vecy1,path)
        vecs_elem_y.Add(vecy2,path)
        vecs_elem_z.Add(vecz1,path)
        vecs_elem_z.Add(vecz2,path)

pt_c = rs.AddPoint(0,0,0)

# 全点列P,Q作成,出力,代表線分PQ作成
pts_pp_all = DataTree[System.Object]()
pts_qq_all = DataTree[System.Object]()
pts_pp_off_all = DataTree[System.Object]()
pts_qq_off_all = DataTree[System.Object]()
lines_pq_all = DataTree[System.Object]()

for ic in range(lcount):
    path = GH_Path(0, ic)
    _xp = rs.coerce3dpoint(pts_pp[ic]).X
    _zp = rs.coerce3dpoint(pts_pp[ic]).Z
    _xq = rs.coerce3dpoint(pts_qq[ic]).X
    _zq = rs.coerce3dpoint(pts_qq[ic]).Z

    _xp_off = rs.coerce3dpoint(pts_pp_off[ic]).X
    _zp_off = rs.coerce3dpoint(pts_pp_off[ic]).Z
    _xq_off = rs.coerce3dpoint(pts_qq_off[ic]).X
    _zq_off = rs.coerce3dpoint(pts_qq_off[ic]).Z

    rot = math.pi/nsym
    for ip in range(nsym):
        if ic % 2 == 0:
            pts_pp_all.Add(rs.AddPoint(_xp*math.cos(rot*(2*ip)), \
                        _xp*math.sin(rot*(2*ip)), _zp), path)
            pts_pp_off_all.Add(rs.AddPoint(_xp_off*math.cos(rot*(2*ip)), \
                        _xp_off*math.sin(rot*(2*ip)), _zp_off), path)
        else: # ic % 2 == 1:
            pts_pp_all.Add(rs.AddPoint(_xp*math.cos(rot*(2*ip+1)), \
                        _xp*math.sin(rot*(2*ip+1)), _zp), path)
            pts_pp_off_all.Add(rs.AddPoint(_xp_off*math.cos(rot*(2*ip+1)), \
                        _xp_off*math.sin(rot*(2*ip+1)), _zp_off), path)

        if ic % 2 == 0:
            pts_qq_all.Add(rs.AddPoint(_xq*math.cos(rot*(2*ip+1)), \
                        _xq*math.sin(rot*(2*ip+1)), _zq), path)
            pts_qq_off_all.Add(rs.AddPoint(_xq_off*math.cos(rot*(2*ip+1)), \
                        _xq_off*math.sin(rot*(2*ip+1)), _zq_off), path)
        else: # ic % 2 == 1:
            pts_qq_all.Add(rs.AddPoint(_xq*math.cos(rot*2*(ip+1)), \
                        _xq*math.sin(rot*2*(ip+1)), _zq), path)
            pts_qq_off_all.Add(rs.AddPoint(_xq_off*math.cos(rot*2*(ip+1)), \
                        _xq_off*math.sin(rot*2*(ip+1)), _zq_off), path)
    # 部材線作成
    if ic % 2 == 0:
        line1 = (_xp_off*math.cos(rot*(0)),_xp_off*math.sin(rot*(0)),_zp_off), \
                (_xq_off*math.cos(rot*(1)),_xq_off*math.sin(rot*(1)),_zq_off)
    else: # ic % 2 == 1:
        line1 = (_xp_off*math.cos(rot*(1)),_xp_off*math.sin(rot*(1)),_zp_off), \
                (_xq_off*math.cos(rot*(0)),_xq_off*math.sin(rot*(0)),_zq_off)
    line2 = rs.LineTransform(line1, xform_mirror)
    lines_pq_all.Add(rs.AddLine(line1[0],line1[1]), path)
    lines_pq_all.Add(rs.AddLine(line2[0],line2[1]), path)
    for ip in range(1,nsym):
        line1 = rs.LineTransform(line1, xform_rotate)
        line2 = rs.LineTransform(line2, xform_rotate)
        lines_pq_all.Add(rs.AddLine(line1[0],line1[1]), path)
        lines_pq_all.Add(rs.AddLine(line2[0],line2[1]), path)

# ヒンジ軸作成
vecs_hinge_pp = DataTree[System.Object]()
vec_hinge_pp = [rg.Vector3d( \
 math.cos(num_beta_pp[ic]),0,math.sin(num_beta_pp[ic])) for ic in range(lcount)]
for ic in range(lcount):
    path = GH_Path(0, ic)
    if ic % 2 == 0:
        vec_h = vec_hinge_pp[ic]
    else: # ic % 2 == 1:
        vec_h = rs.VectorTransform(vec_hinge_pp[ic],xform_rotate_h) 
    vecs_hinge_pp.Add(vec_h, path)
    for ip in range(1,nsym):
        vec_h = rs.VectorTransform(vec_h, xform_rotate)
        vecs_hinge_pp.Add(vec_h, path)

vecs_hinge_qq = DataTree[System.Object]()
vec_hinge_qq = [rg.Vector3d( \
 math.cos(num_beta_qq[ic]),0,math.sin(num_beta_qq[ic])) for ic in range(lcount)]
for ic in range(lcount):
    path = GH_Path(0, ic)
    if ic % 2 == 0:
        vec_h = rs.VectorTransform(vec_hinge_qq[ic],xform_rotate_h) 
    else: # ic % 2 == 1:
        vec_h = rs.VectorTransform(vec_hinge_qq[ic],xform_rotate) 
    vecs_hinge_qq.Add(vec_h, path)
    for ip in range(1,nsym):
        vec_h = rs.VectorTransform(vec_h, xform_rotate)
        vecs_hinge_qq.Add(vec_h, path)

vecs_hinge_cnn = DataTree[System.Object]()
vec_hinge_cnn = rg.Vector3d(0,1,0)
for ic in range(lcount-1):
    path = GH_Path(0, ic)
    if ic % 2 == 0:
        vec_h = rs.VectorTransform(vec_hinge_cnn,xform_rotate_h)
    else: #if % 2 == 1:
        vec_h = rs.VectorTransform(vec_hinge_cnn,xform_rotate)
    vecs_hinge_cnn.Add(vec_h, path)
    for ip in range(1,nsym):
        vec_h = rs.VectorTransform(vec_h, xform_rotate)
        vecs_hinge_cnn.Add(vec_h, path)

# オフセット軸作成
vecs_off_pp = DataTree[System.Object]()
vec_off_pp = [rg.Vector3d(math.cos(num_offangle_pp[ic]),0,\
    math.sin(num_offangle_pp[ic])) for ic in range(lcount)]
for ic in range(lcount):
    path = GH_Path(0, ic)
    if ic % 2 == 0:
        vec_o = vec_off_pp[ic]
    else: # ic % 2 == 1:
        vec_o = rs.VectorTransform(vec_off_pp[ic],xform_rotate_h) 
    vecs_off_pp.Add(vec_o, path)
    for ip in range(1,nsym):
        vec_o = rs.VectorTransform(vec_o, xform_rotate)
        vecs_off_pp.Add(vec_o, path)

vecs_off_qq = DataTree[System.Object]()
vec_off_qq = [rg.Vector3d(math.cos(num_offangle_qq[ic]),0,\
    math.sin(num_offangle_qq[ic])) for ic in range(lcount)]
for ic in range(lcount):
    path = GH_Path(0, ic)
    if ic % 2 == 0:
        vec_o = rs.VectorTransform(vec_off_qq[ic],xform_rotate_h) 
    else: # ic % 2 == 1:
        vec_o = rs.VectorTransform(vec_off_qq[ic],xform_rotate) 
    vecs_off_qq.Add(vec_o, path)
    for ip in range(1,nsym):
        vec_o = rs.VectorTransform(vec_o, xform_rotate)
        vecs_off_qq.Add(vec_o, path)

