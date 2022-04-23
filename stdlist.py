import numpy as np
import scipy.interpolate as ipl
import scipy.optimize as opt
import matplotlib.pyplot as plt
import pandas as pd


PI = np.pi

class StdList:
    '''
        Crawford1974に基づく基準量の計算のクラス
        n_sym : 対称性 デフォルト=3
        num_div : 補間のため片側何分割するか デフォルトは1000
    '''
    # ---------- コンストラクタ
   

    def __init__(self, n_sym=3, num_div_h=1000):
        self.n_sym = n_sym
        self.num_div_h = num_div_h
        self.num_div = num_div_h*2+1

        self.num_t = np.delete(np.linspace(-1,1,self.num_div),[0,self.num_div-1])
        self.num_t = np.append(-0.99999999,self.num_t) #数値計算上の処理（0/0を避ける）
        self.num_t = np.append(self.num_t,0.99999999) #数値計算上の処理（0/0を避ける）
        self.num_t[num_div_h]=0.00000001 #数値計算上の処理（0/0を避ける）

        # calc1 で計算
        self.num_rr = []
        self.num_vv = []
        self.num_ss = []
        self.num_beta_pp = []
        self.num_beta_qq = []
        self.theta_p = []
        self.theta_r = []
        self.theta_y = []

        # calc2 で計算
        self.num_bb = []
        self.num_tt = []
        self.num_phi = []
        self.num_psi = []
        self.num_kappa = []

    def calc(self):
        ''' SCHF の計算一式を全て実行 '''
        # calc_1
        self.calc_1()
        # calc_2
        self.calc_2()
        # calc_ipl
        self.calc_ipl()


    # ---------- calc_1 
        
    def calc_1(self):
        ''' SCHF の 計算1（前半）Crawfordの式に基づく計算 '''
        for _t in self.num_t[0:self.num_div]:
            _theta_p, _theta_r, _theta_y, _num_rr, _num_ss, _num_vv, _num_beta_pp, _num_beta_qq \
             = self.return_param(_t) 
            #  配列への格納
            self.theta_p.append(_theta_p)
            self.theta_r.append(_theta_r)
            self.theta_y.append(_theta_y)
            self.num_rr.append(_num_rr)
            self.num_ss.append(_num_ss)
            self.num_vv.append(_num_vv)
            self.num_beta_pp.append(_num_beta_pp)
            self.num_beta_qq.append(_num_beta_qq)
        
        #  ndarrayに変換
        self.theta_p=np.array(self.theta_p)
        self.theta_r=np.array(self.theta_r)
        self.theta_y=np.array(self.theta_y)
        self.num_rr=np.array(self.num_rr)
        self.num_ss=np.array(self.num_ss)
        self.num_vv=np.array(self.num_vv)
        self.num_beta_pp=np.array(self.num_beta_pp)
        self.num_beta_qq=np.array(self.num_beta_qq)
            
        # Rが最大となるidxを取得し、その前後で分けたR,tの配列を作成
        # （これらは数式で厳密に定義可能ため、追って修正する）
        self.idx_rr_max = np.argmax(self.num_rr)
        self.rr_max = self.num_rr[self.idx_rr_max]
        self.t_at_rr_max = self.num_t[self.idx_rr_max]
        
        self.num_rr_neg = self.num_rr[0:self.idx_rr_max]
        self.num_t_neg = self.num_t[0:self.idx_rr_max]

        self.num_rr_pos = self.num_rr[self.idx_rr_max+1:]
        self.num_t_pos = self.num_t[self.idx_rr_max+1:]

    def return_param(self, _t):
        ''' t に対するSCHFの幾何学パラメータを返す(calc_1)'''
        #theta_p,theta_r,theta_yの計算
        if _t<=0 :
            _theta_p = - PI/2 * _t
            _theta_r = self.return_theta_r(self.n_sym, _theta_p, False)
            #最後の引数はSCHF-NegativeならFalse, -PositiveならTrueにする
        else :
            _theta_p = PI/2 * _t
            _theta_r = self.return_theta_r(self.n_sym, _theta_p, True)

        _theta_y = self.return_theta_y(self.n_sym, _theta_p, _theta_r)
    
        #R,S,Vの計算
        _num_rr,_num_ss,_num_vv \
          = self.return_rsv(self.n_sym, _theta_p, _theta_r, _theta_y)
        #betaP,betaQの計算
        _num_beta_pp,_num_beta_qq \
          = self.return_beta(self.n_sym,_theta_p,_theta_r,_theta_y)

        return _theta_p, _theta_r, _theta_y, _num_rr, _num_ss, _num_vv, _num_beta_pp, _num_beta_qq
        
    def return_beta(self, _n_sym, _theta_p, _theta_r, _theta_y):
        dlt=PI/(2*_n_sym)
        c_dl=np.cos(dlt)
        s_dl=np.sin(dlt)
        c_tp=np.cos(_theta_p)
        s_tp=np.sin(_theta_p)
        c_tr=np.cos(_theta_r)
        s_tr=np.sin(_theta_r)
        c_ty=np.cos(_theta_y)
        s_ty=np.sin(_theta_y)
        
        alpha1=np.arctan(-s_dl/c_dl**2)
        alpha2=-alpha1
        
        c_al1=np.cos(alpha1)
        s_al1=np.sin(alpha1)
        c_al2=np.cos(alpha2)
        s_al2=np.sin(alpha2)
        
        lx1=(s_dl*c_ty*c_tp - c_dl*s_ty*c_tr + c_dl*c_ty*s_tp*s_tr)*s_al1 \
          +(s_ty*s_tr + c_ty*s_tp*c_tr)*c_al1
        ly1=(s_dl*s_ty*c_tp + c_dl*c_ty*c_tr + c_dl*s_ty*s_tp*s_tr)*s_al1 \
          +(-c_ty*s_tr + s_ty*s_tp*c_tr)*c_al1
        lz1=(-s_dl*s_tp + c_dl*c_tp*s_tr)*s_al1 + c_tp*c_tr*c_al1
        lh1=lx1*s_dl + ly1*c_dl
        _beta_pp=np.arctan2(lz1,lh1)

        lx2=(-s_dl*c_ty*c_tp - c_dl*s_ty*c_tr + c_dl*c_ty*s_tp*s_tr)*s_al2 \
          +(s_ty*s_tr + c_ty*s_tp*c_tr)*c_al2
        ly2=(-s_dl*s_ty*c_tp + c_dl*c_ty*c_tr + c_dl*s_ty*s_tp*s_tr)*s_al2 \
          +(-c_ty*s_tr + s_ty*s_tp*c_tr)*c_al2
        lz2=(s_dl*s_tp + c_dl*c_tp*s_tr)*s_al2 + c_tp*c_tr*c_al2
        lh2=-lx2*s_dl + ly2*c_dl
        _beta_qq=np.arctan2(lz2,lh2)

        return _beta_pp, _beta_qq
        
    def return_rsv(self, _n_sym, _theta_p, _theta_r, _theta_y):
        dlt=PI/(2*_n_sym)
        s_len=2*np.sin(dlt) # 正多角形の辺長
        cos_tp=np.cos(_theta_p)
        sin_tp=np.sin(_theta_p)
        cos_dmty=np.cos(dlt-_theta_y)
        sin_dmty=np.sin(dlt-_theta_y)
        cos_dpty=np.cos(dlt+_theta_y)
        sin_dpty=np.sin(dlt+_theta_y)
        
        tmp_a = np.abs(cos_dmty*sin_dmty*cos_dpty)
        tmp_b = cos_dmty**2 * sin_dpty
        tmp_c = sin_dmty**2 - sin_dpty**2

        _rr1=s_len*cos_tp*( tmp_a-tmp_b)/tmp_c
        _ss1=(s_len*cos_tp-_rr1*sin_dpty)/sin_dmty
        _rr2=s_len*cos_tp*(-tmp_a-tmp_b)/tmp_c
        _ss2=(s_len*cos_tp-_rr2*sin_dpty)/sin_dmty
        if _rr1*_ss1 >=0 and _rr2*_ss2 <= 0:
            _rr=_rr1
            _ss=_ss1
        elif _rr1*_ss1 <=0 and _rr2*_ss2 >=0:
            _rr=_rr2
            _ss=_ss2
        else:
            print("error when _rr1,_ss1,_rr2,_ss2=",_rr1,_ss1,_rr2,_ss2)
            print("tmp_a,tmp_b,tmp_c=",tmp_a,tmp_b,tmp_c)
            print("_theta_p,_theta_r,_theta_y,_n_sym",_theta_p,_theta_r,_theta_y,_n_sym)

        _vv=s_len*sin_tp
        
        return _rr, _ss, _vv
        
    def return_theta_r(self, _n_sym, _theta_p, boo):
        ''' theta_r を計算して返す関数 boo=FalseならSCHF-Negative, boo=TrueならSCHF-Positiveとして返す'''
        dlt=PI/(2*_n_sym)
        cos_dlt=np.cos(dlt)
        sin_dlt=np.sin(dlt)
        cos_tp=np.cos(_theta_p)
        sin_tp=np.sin(_theta_p)
        tmp_a = cos_dlt**2 * cos_tp **2
        tmp_b = sin_dlt**2 * (cos_dlt**2*(2*sin_tp-1)+sin_dlt**2)*cos_tp
        tmp_c = -(1-sin_tp)*(1-2*sin_dlt**2 * cos_dlt**2 + sin_dlt**2 * sin_tp)
        tmp1, cos_theta_r = self.solve_qe(tmp_a,tmp_b,tmp_c)
        # 2次方程式の解のうち、√Dの係数がプラスの方のみ採用
        if boo == False:
            _theta_r = np.arccos(cos_theta_r)
        else:
            _theta_r = -np.arccos(cos_theta_r)
        if np.isnan(_theta_r) == True and np.abs(_theta_p) <= 0.0000001:
            _theta_r = 0
        return _theta_r    

    def return_theta_y(self, _n_sym, _theta_p, _theta_r):
        ''' theta_y を計算して返す関数'''
        dlt=PI/(2*_n_sym)
        sin_tp = np.sin(_theta_p)
        cos_tp = np.cos(_theta_p)
        sin_tr = np.sin(_theta_r)
        cos_tr = np.cos(_theta_r)
        tan_theta_y = - sin_tr / (cos_tr + (np.tan(dlt)**2) * (cos_tp/(1-sin_tp)))
        _theta_y = np.arctan(tan_theta_y)
        return _theta_y
    
    def solve_qe(self, _tmp_a, _tmp_b, _tmp_c):
        root_d = np.sqrt(_tmp_b**2 - 4*_tmp_a*_tmp_c)
        return (-_tmp_b - root_d)/(2*_tmp_a), (-_tmp_b + root_d)/(2*_tmp_a)

    # ---------- calc2 

    def calc_2(self):    
        ''' SCHF の 計算2（後半）phi,psi,bb,ttなどの計算 '''

        # 底辺B,長辺T,頂角Phi,底角Psiの計算
        self.num_bb, self.num_tt, self.num_phi, self.num_psi \
         = self.return_bb_tt_phi_psi(self.n_sym, self.num_rr, self.num_vv, self.num_ss)

        # kappaの計算
        self.num_kappa = self.return_kappa(self.num_rr, self.num_vv, self.num_ss)
        
    def return_kappa(self, _num_rr, _num_vv, _num_ss):
        ''' 角度kappaの計算 calc2で使用 '''
        tmp1=_num_ss-_num_rr
        tmp2=tmp1/_num_vv
        return np.arctan2(_num_vv, _num_ss-_num_rr)

    def return_bb_tt_phi_psi(self,_n_sym, _num_rr, _num_vv, _num_ss):
        ''' 底辺B,頂辺T,頂角Phi,底角Psiの計算 calc2で使用 '''
        tht=PI/_n_sym
        _ll=2*np.sin(tht/2)
        _bb=2*_num_rr*np.sin(tht)
        _tt=2*_num_ss*np.sin(tht)
        _phi=2*np.arcsin(_bb/(2*_ll))
        _psi=2*np.arcsin(_tt/(2*_ll))
        return _bb, _tt, _phi, _psi
        
        
    # ---------- 関数補間とGベクトルの作成
    def calc_ipl(self):
        ''' パラメータtに関する各諸元関数の補間 scipy.interpolateを使用 '''
        self.ipl_rr_t = ipl.interp1d(self.num_t, self.num_rr, kind='cubic',bounds_error=False)
        self.ipl_vv_t = ipl.interp1d(self.num_t, self.num_vv, kind='cubic',bounds_error=False)
        self.ipl_ss_t = ipl.interp1d(self.num_t, self.num_ss, kind='cubic',bounds_error=False)
        self.ipl_beta_pp_t = ipl.interp1d(self.num_t, self.num_beta_pp, kind='cubic',bounds_error=False)
        self.ipl_beta_qq_t = ipl.interp1d(self.num_t, self.num_beta_qq, kind='cubic',bounds_error=False)

        self.ipl_bb_t = ipl.interp1d(self.num_t, self.num_bb,  kind='cubic',bounds_error=False)
        self.ipl_tt_t = ipl.interp1d(self.num_t, self.num_tt,  kind='cubic',bounds_error=False)
        self.ipl_phi_t = ipl.interp1d(self.num_t, self.num_phi,  kind='cubic',bounds_error=False)
        self.ipl_psi_t = ipl.interp1d(self.num_t, self.num_psi,  kind='cubic',bounds_error=False)
        self.ipl_kappa_t = ipl.interp1d(self.num_t, self.num_kappa,  kind='cubic',bounds_error=False)
        
        self.ipl_theta_r_t = ipl.interp1d(self.num_t, self.theta_r,  kind='cubic',bounds_error=False)
        self.ipl_theta_y_t = ipl.interp1d(self.num_t, self.theta_y,  kind='cubic',bounds_error=False)
        
        ''' Rからtへの逆写像を返す関数の作成 '''
        self.ipl_t_rr_pos = ipl.interp1d(self.num_rr_pos, self.num_t_pos, kind='cubic',bounds_error=True)
        self.ipl_t_rr_neg = ipl.interp1d(self.num_rr_neg, self.num_t_neg, kind='cubic',bounds_error=True)

    def vecff(self,_t):
        ''' Fベクトル 2成分を返す '''
        return self.ipl_ss_t(_t) - self.ipl_rr_t(_t), self.ipl_vv_t(_t)
        
    def vecgg(self,_t):
        ''' Gベクトル 2成分を返す '''
        return (self.ipl_ss_t(_t) - self.ipl_rr_t(_t))/self.ipl_rr_t(_t), self.ipl_vv_t(_t)/self.ipl_rr_t(_t)


    # ---------- 以降はメソッド -------------

    def calc_crawford_elem_vec_t(self, _t):
        # 部材座標軸ベクトルを返す
        # 部材はP1Q1（P1はX軸上でY=Z=0)とする。
        
        _theta_p = np.abs(PI/2*_t)
        _theta_r = self.ipl_theta_r_t(_t)
        _theta_y = self.ipl_theta_y_t(_t)
        
        tmp_x = np.zeros((3))
        tmp_y = np.zeros((3))
        tmp_z = np.zeros((3))

        c_p = np.cos(_theta_p)
        s_p = np.sin(_theta_p)
        c_r = np.cos(_theta_r)
        s_r = np.sin(_theta_r)
        c_y = np.cos(_theta_y)
        s_y = np.sin(_theta_y)
        
        tmp_x = np.array([c_y*c_p, s_y*c_p, -s_p])
        tmp_y = np.array([-s_y*c_r+c_y*s_p*s_r, c_y*c_r+s_y*s_p*s_r, c_p*s_r])
        tmp_z = np.array([s_y*s_r + c_y*s_p*c_r, -c_y*s_r + s_y*s_p*c_r, c_p*c_r])

        # Crawford の全体座標系から採用している全体座標系への変換
        tmp_x[0], tmp_y[0], tmp_z[0] = -tmp_x[0], -tmp_y[0], -tmp_z[0]
        tmp_x[1], tmp_y[1], tmp_z[1] = -tmp_x[1], -tmp_y[1], -tmp_z[1]
        
        tmp_x = -tmp_x
        tmp_y = -tmp_y
        
        # 偏角0 - PI/n_sym 間の部材用に回転
        # 回転角 PI/2 + PI/(2*n_sym)
        _tht = PI/2 + PI/(2*self.n_sym)
        _R = np.array([[np.cos(_tht), -np.sin(_tht), 0],
                       [np.sin(_tht),  np.cos(_tht), 0],
                       [0, 0, 1]])

        tmp_x = np.dot(_R, tmp_x)
        tmp_y = np.dot(_R, tmp_y)
        tmp_z = np.dot(_R, tmp_z)
        
        return tmp_x, tmp_y, tmp_z

    def calc_xyz(self,_t):
        # 環状の各節点座標を返す
        # xx : 節点P_1,Q_1,P_2,Q_2,...,P_N,Q_Nのtでのx座標
        # yy : 節点P_1,Q_1,P_2,Q_2,...,P_N,Q_Nのtでのy座標
        # zz : 節点P_1,Q_1,P_2,Q_2,...,P_N,Q_Nのtでのz座標

        _delta = PI/(self.n_sym * 2)

        # X座標
        xx = np.array(self.ipl_rr_t(_t)*np.cos(_delta*2*(2*0)))
        xx = np.vstack((xx, np.array(self.ipl_ss_t(_t)*np.cos(_delta*2*(2*0+1)))))
        for i in range(1,self.n_sym):
            xx = np.vstack((xx,np.array(self.ipl_rr_t(_t)*np.cos(_delta*2*(2*i)))))
            xx = np.vstack((xx,np.array(self.ipl_ss_t(_t)*np.cos(_delta*2*(2*i+1)))))
        xx = np.vstack((xx, self.ipl_rr_t(_t)*np.cos(_delta*2*(2*0))))

        # Y座標
        yy = np.array(self.ipl_rr_t(_t)*np.sin(_delta*2*(2*0)))
        yy = np.vstack((yy, np.array(self.ipl_ss_t(_t)*np.sin(_delta*2*(2*0+1)))))
        for i in range(1,self.n_sym):
            yy = np.vstack((yy,np.array(self.ipl_rr_t(_t)*np.sin(_delta*2*(2*i)))))
            yy = np.vstack((yy,np.array(self.ipl_ss_t(_t)*np.sin(_delta*2*(2*i+1)))))
        yy = np.vstack((yy,self.ipl_rr_t(_t)*np.sin(_delta*2*(2*0))))

        # Z座標
        zz = np.zeros(_t.shape[0])
        zz = np.vstack((zz, np.array(self.ipl_vv_t(_t))))
        for i in range(1,self.n_sym):
            zz = np.vstack((zz,np.zeros(_t.shape[0])))
            zz = np.vstack((zz,np.array(self.ipl_vv_t(_t))))
        zz = np.vstack((zz,np.zeros(_t.shape[0])))

        return xx,yy,zz

    def calc_hvec_xyz(self,_t):
        # 環状の各節点のヒンジベクトルを返す
        # hvec_xx : 節点P_1,Q_1,P_2,Q_2,...,P_N,Q_Nのヒンジベクトルのtでのx成分
        # hvec_yy : 節点P_1,Q_1,P_2,Q_2,...,P_N,Q_Nのヒンジベクトルのtでのy成分
        # hvec_zz : 節点P_1,Q_1,P_2,Q_2,...,P_N,Q_Nのヒンジベクトルのtでのz成分

        _delta = PI/(self.n_sym * 2)

        beta_pp = self.ipl_beta_pp_t(_t)
        beta_qq = self.ipl_beta_qq_t(_t)
        hvec_hh_pp = np.cos(beta_pp)
        hvec_vv_pp = np.sin(beta_pp)
        hvec_hh_qq = np.cos(beta_qq)
        hvec_vv_qq = np.sin(beta_qq)

        # x成分
        hvec_xx = np.array(hvec_hh_pp*np.cos(_delta*2*(2*0)))
        hvec_xx = np.vstack((hvec_xx, hvec_hh_qq*np.cos(_delta*2*(2*0+1))))
        for i in range(1,self.n_sym):
            hvec_xx = np.vstack((hvec_xx,hvec_hh_pp*np.cos(_delta*2*(2*i))))
            hvec_xx = np.vstack((hvec_xx,hvec_hh_qq*np.cos(_delta*2*(2*i+1))))
        hvec_xx = np.vstack((hvec_xx, hvec_hh_pp*np.cos(_delta*2*(2*0))))

        # y成分
        hvec_yy = np.array(hvec_hh_pp*np.sin(_delta*2*(2*0)))
        hvec_yy = np.vstack((hvec_yy, hvec_hh_qq*np.sin(_delta*2*(2*0+1))))
        for i in range(1,self.n_sym):
            hvec_yy = np.vstack((hvec_yy,hvec_hh_pp*np.sin(_delta*2*(2*i))))
            hvec_yy = np.vstack((hvec_yy,hvec_hh_qq*np.sin(_delta*2*(2*i+1))))
        hvec_yy = np.vstack((hvec_yy, hvec_hh_pp*np.sin(_delta*2*(2*0))))

        # z成分
        hvec_zz = np.array(hvec_vv_pp)
        hvec_zz = np.vstack((hvec_zz, np.array(hvec_vv_qq)))
        for i in range(1,self.n_sym):
            hvec_zz = np.vstack((hvec_zz,np.array(hvec_vv_pp)))
            hvec_zz = np.vstack((hvec_zz,np.array(hvec_vv_qq)))
        hvec_zz = np.vstack((hvec_zz,np.array(hvec_vv_pp)))

        return hvec_xx,hvec_yy,hvec_zz



    def draw_graphs(self,legend=True,outfile=False,path=""):
        # matplotlibでグラフを描画する

        plt.clf()

        spines = 1.5 # 枠の太さ

        fig=plt.figure(figsize=(10,15))

        ax1=fig.add_subplot(3,2,1)
        ax1.plot(self.num_t, self.num_rr, '-', label='num_rr', markersize=0.1)
        ax1.plot(self.num_t, self.num_ss, '-', label='num_ss', markersize=0.1)
        ax1.plot(self.num_t, self.num_vv, '-', label='num_vv', markersize=0.1)
        ax1.legend(loc='best')
        ax1.set_xlabel('parameter t')
        ax1.set_ylabel('**')
        ax1.grid('on')
        ax1.spines["top"].set_linewidth(spines)
        ax1.spines["left"].set_linewidth(spines)
        ax1.spines["bottom"].set_linewidth(spines)
        ax1.spines["right"].set_linewidth(spines)
        ax1.set_title('R,S,V')

        if legend == False: ax1.get_legend().remove()

        ax2=fig.add_subplot(3,2,2)
        ax2.plot(self.vecff(self.num_t)[0], self.vecff(self.num_t)[1], '-', label='F vec', markersize=0.1)
        for i in [0,1,2,3,4,5,6,7,8,9]:
            _idx = self.num_div * i // 10
            ax2.plot([0,self.vecff(self.num_t[_idx])[0]], [0,self.vecff(self.num_t[_idx])[1]], '-o', markersize=3, color='gray')
        ax2.legend(loc='best')
        ax2.set_xlabel('parameter t')
        ax2.set_ylabel('**')
        ax2.set_ylim(0,1)
        ax2.grid('on')
        ax2.spines["top"].set_linewidth(spines)
        ax2.spines["left"].set_linewidth(spines)
        ax2.spines["bottom"].set_linewidth(spines)
        ax2.spines["right"].set_linewidth(spines)
        ax2.set_title('Vector function F')
        if legend == False: ax2.get_legend().remove()

        ax3=fig.add_subplot(3,2,3)
        ax3.plot(self.vecgg(self.num_t)[0], self.vecgg(self.num_t)[1], '-', label='G vec', markersize=0.1)
        for i in [1,2,3,4,5,6,7,8,9]:
            _idx = self.num_div * i // 10
            ax3.plot([0,self.vecgg(self.num_t[_idx])[0]], [0,self.vecgg(self.num_t[_idx])[1]], '-o', markersize=3, color='gray')
        ax3.legend(loc='best')
        ax3.set_xlabel('parameter t')
        ax3.set_xlim([-0.9,0.9])
        ax3.set_ylabel('**')
        ax3.set_ylim(0,6)
        ax3.grid('on')
        ax3.spines["top"].set_linewidth(spines)
        ax3.spines["left"].set_linewidth(spines)
        ax3.spines["bottom"].set_linewidth(spines)
        ax3.spines["right"].set_linewidth(spines)
        ax3.set_title('Vector function G')
        if legend == False: ax3.get_legend().remove()

        ax4=fig.add_subplot(3,2,4)
        ax4.plot(self.num_t, self.num_phi, '-', label='num_phi', markersize=0.1)
        ax4.plot(self.num_t, self.num_psi, '-', label='num_psi', markersize=0.1)
        ax4.plot(self.num_t, self.num_kappa, '-', label='num_kappa', markersize=0.1)
        ax4.legend(loc='best')
        ax4.set_xlabel('parameter t')
        ax4.set_ylabel('**')
        ax4.grid('on')
        ax4.spines["top"].set_linewidth(spines)
        ax4.spines["left"].set_linewidth(spines)
        ax4.spines["bottom"].set_linewidth(spines)
        ax4.spines["right"].set_linewidth(spines)
        ax4.set_title('phi, psi, kappa')
        if legend == False: ax4.get_legend().remove()

        ax5=fig.add_subplot(3,2,5)
        ax5.plot(self.num_t, self.num_beta_pp, '-', label='num_beta_pp', markersize=0.1)
        ax5.plot(self.num_t, self.num_beta_qq, '-', label='num_beta_qq', markersize=0.1)
        ax5.legend(loc='best')
        ax5.set_xlabel('parameter t')
        ax5.set_ylabel('**')
        ax5.grid('on')
        ax5.spines["top"].set_linewidth(spines)
        ax5.spines["left"].set_linewidth(spines)
        ax5.spines["bottom"].set_linewidth(spines)
        ax5.spines["right"].set_linewidth(spines)
        ax5.set_title('beta_P, beta_Q')
        if legend == False: ax5.get_legend().remove()

        ax6=fig.add_subplot(3,2,6)
        ax6.plot(self.num_rr_neg, self.ipl_t_rr_neg(self.num_rr_neg), '-', label='Rinv_neg', markersize=0.1)
        ax6.plot(self.num_rr_pos, self.ipl_t_rr_pos(self.num_rr_pos), '-', label='Rinv_pos', markersize=0.1)
        ax6.legend(loc='best')
        ax6.set_xlabel('R')
        ax6.set_ylabel('parameter t')
        ax6.grid('on')
        ax6.spines["top"].set_linewidth(spines)
        ax6.spines["left"].set_linewidth(spines)
        ax6.spines["bottom"].set_linewidth(spines)
        ax6.spines["right"].set_linewidth(spines)
        ax6.set_title('R inverse function')
        if legend == False: ax6.get_legend().remove()

        if outfile == True:
            plt.savefig(path,edgecolor="coral")   # プロットしたグラフをファイルsin.pngに保存する
            plt.show()