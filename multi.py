from multiprocessing import dummy
import numpy as np
import scipy.interpolate as ipl
import scipy.optimize as opt
import matplotlib.pyplot as plt
import pandas as pd
import time

PI = np.pi

class MultiLayerHF:
    '''
        MultiLayerHingeFrameのクラス
    '''
    # ---------- コンストラクタ
   
    def __init__(self, stdlist, pts_gen, layer_length_min=0.1, h_off=0, div_u=1001):
        '''コンストラクタ'''
        # stdlist : stdlistクラスのインスタンス
        # pts_gen : 母関数定義用の3次元座標点 (4点以上) 現状は軸はグローバルZ軸のみ対応
        # layer_length_min : 生成層の最小RZ平面上長さ (デフォルトは0.1mとする)
        # h_off : ヒンジ部オフセット量（デフォルト0）
        # div_u : パラメータ u (0<=u<=1)の分割数 (デフォルトは1001とする)
        
        self.stdlist = stdlist
        self.pts_gen = pts_gen
        self.layer_length_min = layer_length_min
        self.div_u = div_u
        self.h_off = h_off
        
        # 取得したstdlistからnsym,vecgg等を取得
        self.n_sym = stdlist.n_sym
        self.vecgg = stdlist.vecgg
        self.ipl_beta_pp_t = stdlist.ipl_beta_pp_t
        self.ipl_beta_qq_t = stdlist.ipl_beta_qq_t
        # 母線定義用補間関数R(Z)の定義
        self.gtr_rr_zz=ipl.interp1d(self.pts_gen[:,2], self.pts_gen[:,0], kind='cubic')
        # Zの範囲（最下点座標と最上点座標）取得
        self.zz_max=max(pts_gen[:,2])
        self.zz_min=min(pts_gen[:,2])
        # parameter u 定義
        self.num_u = np.linspace(0, 1, self.div_u)
        # 層数
        self.lc = 0
        
        # 各層特性値格納用リスト
        # self.list_kl : 各層の拡大比kl
        # self.list_t0 : 各層の初期状態でのt(u=0)
        # self.list_t_u : 各層のt(u)を格納する2次元配列(lc x div_u)
        # self.list_coordr_pp : 各層のustepに対する下部節点RZ平面内R座標
        # self.list_coordz_pp : 各層のustepに対する下部節点RZ平面内Z座標
        # self.list_coordr_qq : 同　上部節点
        # self.list_coordz_qq : 同　上部節点
        # self.list_beta_pp : 各層のustepに対する下部節点角度β
        # self.list_beta_qq : 各層のustepに対する上部節点角度β
        # self.list_phi : 
        # self.list_psi : 
        # self.list_kappa : 
        # list_elem_coord_x : 各層の各ustepに対する偏角0 - PI/n_sym 間の部材座標軸ベクトル（材軸方向）
        # list_elem_coord_y : 各層の各ustepに対する偏角0 - PI/n_sym 間の部材座標軸ベクトル（断面軸方向1）
        # list_elem_coord_z : 各層の各ustepに対する偏角0 - PI/n_sym 間の部材座標軸ベクトル（断面軸方向2）
        
        
    # ---------- calc 1 
    def calc_1(self):
        ''' 全ての層の初期状態(ti0とki)の計算 '''
        print("<<--- START calc_1 --->>")
        # 時間計測
        start_time = time.process_time()

        self.list_kl = []
        self.list_t0 = []
        
        # 各層でのループ
        zz_prev = self.zz_min #ループ初期点定義
        rr_prev = self.gtr_rr_zz(zz_prev) #ループ初期点定義
        _lc = 0 #層数カウンタ
        _h_off = self.h_off # オフセット量（ここでは全層同値とする）

        _flag = True #ループ継続判定
        while _flag:
            # 次の交点の算定 失敗の場合、t0=kl=0, _flag=Falseを返す
            t0, kl, _flag = self.calc_t0_kl(rr_prev,zz_prev,_h_off)

            # 上点候補の計算（オフセット処理を考慮）
            _d_pp, _d_qq = self.return_offvecs_dpp_dqq(t0,_h_off)
            rr_next, zz_next = self.return_next_rr_and_zz(t0,rr_prev,zz_prev,_d_pp,_d_qq)
            _length_r = rr_next - rr_prev
            _length_z = zz_next - zz_prev

            # _layer_length : 生成層のRZ平面上長さ
            _layer_length = np.sqrt((_length_r)**2 + (_length_z)**2)
 
            if _layer_length < self.layer_length_min :
                print('new layer length too short, not made')
                _flag = False
            # 生成層の長さが規定値未満なら新しく層を追加しない

            if _flag : # 層追加条件を満たす場合
                _lc += 1

                # 上点の計算
#                zz_tmp = zz_prev + rr_prev_off * self.vecgg(t0)[1]
#                rr_tmp = rr_prev + rr_prev * self.vecgg(t0)[0]

                #  リストへの格納
                self.list_kl.append(float(kl))
                self.list_t0.append(float(t0))
                
                # 次層の下点の更新
                zz_prev = zz_next
                rr_prev = rr_next

            else :
                self.lc = _lc  # 層数確定
        self.list_kl = np.array(self.list_kl)
        self.list_t0 = np.array(self.list_t0)
        print(" -- result --")
        print(" list_t0")
        print(self.list_t0)
        print(" list_kl")
        print(self.list_kl)

        # 終了
        end_time = time.process_time()

        print("<<--- END calc_1 --->>")

        # 処理時間出力(秒)
        proc_time = end_time - start_time
        print('calc_1 time = ',proc_time)

        return

    def calc_t0_kl(self, _rr_prev, _zz_prev, _h_off):
        ''' 各層の指定母線へのフィッティングと、そのときの各層のti(u=0),klの計算 '''
        
        # Zの最大値に対するtの取得
        _tmin, _tmax, _tnegmax, _tposmin = self.return_tminmax(_rr_prev, _zz_prev, _h_off)
        print('tmin,tmax=',_tmin,_tmax)

        # 母線曲線とベクトルG×R_prevの交点サーチ
        # 探索成功フラグ
        flg_cp_success = False
        flg_cp_success_pos, flg_cp_success_neg = False, False
        
        # Positive 方向の二分法サーチ
#        if self.tmpfunc(np.max([_tmin,0.0001]), _rr_prev, _zz_prev, _h_off) * self.tmpfunc(_tmax, _rr_prev, _zz_prev, _h_off) < 0:
#            # Positive 方向の探索
#            _t0, r = opt.bisect(self.tmpfunc, np.max([_tmin,0.0001]), _tmax, args=(_rr_prev, _zz_prev, _h_off), full_output=True, disp=False)
#            flg_cp_success = r.converged
#        elif self.tmpfunc(_tmin, _rr_prev, _zz_prev, _h_off) * self.tmpfunc(np.min([_tmax,-0.0001]), _rr_prev, _zz_prev, _h_off) < 0:
#            # Negative 方向の探索
#            _t0, r = opt.bisect(self.tmpfunc, _tmin, np.min([_tmax,-0.0001]), args=(_rr_prev, _zz_prev, _h_off), full_output=True, disp=False)
#            flg_cp_success = r.converged

        if self.tmpfunc(np.max([_tmin,0.001]), _rr_prev, _zz_prev, _h_off) * self.tmpfunc(_tmax, _rr_prev, _zz_prev, _h_off) < 0:
            # Positive 方向の探索
#            _t0_pos, r = opt.bisect(self.tmpfunc, _tposmin, _tmax, args=(_rr_prev, _zz_prev, _h_off), full_output=True, disp=False)
#            _t0_pos, r = opt.toms748(self.tmpfunc, _tposmin, _tmax, args=(_rr_prev, _zz_prev, _h_off), full_output=True, disp=False)
#            _t0_pos, r = opt.brentq(self.tmpfunc, _tposmin, _tmax, args=(_rr_prev, _zz_prev, _h_off), full_output=True, disp=False)
            _t0_pos, r = opt.brenth(self.tmpfunc, _tposmin, _tmax, args=(_rr_prev, _zz_prev, _h_off), full_output=True, disp=False)
#            _t0_pos, r = opt.ridder(self.tmpfunc, _tposmin, _tmax, args=(_rr_prev, _zz_prev, _h_off), full_output=True, disp=False)
#            _t0_pos, r = opt.brenth(
#                self.tmpfunc, _tposmin, _tmax, args=(_rr_prev, _zz_prev, _h_off), full_output=True, disp=False,
#                xtol=1e-8, rtol=1e-5)
            flg_cp_success_pos = r.converged
            print("I got!! positive _t0 =", _t0_pos)
        if self.tmpfunc(_tmin, _rr_prev, _zz_prev, _h_off) * self.tmpfunc(np.min([_tmax,-0.001]), _rr_prev, _zz_prev, _h_off) < 0:
            # Negative 方向の探索
#            _t0_neg, r = opt.bisect(self.tmpfunc, _tmin, _tnegmax, args=(_rr_prev, _zz_prev, _h_off), full_output=True, disp=False)
#            _t0_neg, r = opt.toms748(self.tmpfunc, _tmin, _tnegmax, args=(_rr_prev, _zz_prev, _h_off), full_output=True, disp=False)
#            _t0_neg, r = opt.brentq(self.tmpfunc, _tmin, _tnegmax, args=(_rr_prev, _zz_prev, _h_off), full_output=True, disp=False)
            _t0_neg, r = opt.brenth(self.tmpfunc, _tmin, _tnegmax, args=(_rr_prev, _zz_prev, _h_off), full_output=True, disp=False)
#            _t0_neg, r = opt.ridder(self.tmpfunc, _tmin, _tnegmax, args=(_rr_prev, _zz_prev, _h_off), full_output=True, disp=False)
#            _t0_neg, r = opt.brenth(
#                self.tmpfunc, _tmin, _tnegmax, args=(_rr_prev, _zz_prev, _h_off), full_output=True, disp=False,
#                xtol=1e-8, rtol=1e-5)
            flg_cp_success_neg = r.converged
            print("I got!! negative _t0 =", _t0_neg)

        # 失敗したときのダミー値
        if flg_cp_success_pos == False:
            _t0_pos = 1.0
        if flg_cp_success_neg == False:
            _t0_neg = -1.0

        flg_cp_success = flg_cp_success_pos or flg_cp_success_neg
        print('flg_cp_success,flg_cp_success_pos,flg_cp_success_neg',flg_cp_success,flg_cp_success_pos,flg_cp_success_neg)

        # 成功ならt=t0 の時の当該層のスケール係数kを計算
        if flg_cp_success :
            if np.abs(_t0_pos) <= np.abs(_t0_neg): # 絶対値の小さい方を採用
                _t0 = _t0_pos
            else:
                _t0 = _t0_neg
            _d_pp, _d_qq = self.return_offvecs_dpp_dqq(_t0, _h_off)
            print('decided _t0', _t0)
            _rr_prev_off = _rr_prev + _d_pp[0]
            _kl = _rr_prev_off/self.stdlist.ipl_rr_t(_t0) # 基準環状骨組に対するスケール(オフセット部除く)
        else :
            print("error 001")
            _t0, _kl = 0, 0

        return _t0, _kl, flg_cp_success
    
    def return_tminmax(self, _rr_prev, _zz_prev, _h_off):
        ''' 二分法サーチをするときのtの上下限値（母線のZの範囲と重なる範囲） '''
        #tmin : 取りうるtの最小値(negative)
        #tmax : 取りうるtの最大値(positive)
        #tnegmax : t=-0.001と符号が同じでありつづけるtの最小値(tのnegative側球根探索上限値)
        #tposmax : t=0.001と符号が同じでありつづけるtの最大値(tのpositive側球根探索下限値)

        _tnegmax = -0.001
        _tposmin = 0.001

        func_val_init = self.tmpfunc(-0.001, _rr_prev, _zz_prev, _h_off)
        for _t in np.linspace(-0.001,-1,1000):
            _d_pp, _d_qq = self.return_offvecs_dpp_dqq(_t, _h_off)

            dummy, zz_tmp = self.return_next_rr_and_zz(_t, _rr_prev, _zz_prev, _d_pp, _d_qq)
            if zz_tmp > self.zz_max:
                break #評価関数のz値が母線関数のZの定義域を超えたら止める。
            _tmin = _t

           # _tmin確定条件追加 4/30
            func_val_tmp =self.tmpfunc(_t, _rr_prev, _zz_prev, _h_off)
            if func_val_init * func_val_tmp < 0.0:
                break #評価関数の符号がt=-0.001でのものと逆符号になる最初の点で止める

            _tnegmax = _t

        func_val_init = self.tmpfunc(0.001, _rr_prev, _zz_prev, _h_off)
        for _t in np.linspace(0.001,1,1000):
            _d_pp, _d_qq = self.return_offvecs_dpp_dqq(_t, _h_off)

            dummy, zz_tmp = self.return_next_rr_and_zz(_t, _rr_prev, _zz_prev, _d_pp, _d_qq)
            if zz_tmp > self.zz_max:
                break #評価関数のz値が母線関数のZの定義域を超えたら止める。
            _tmax = _t

           # _tmax確定条件追加 4/30
            func_val_tmp =self.tmpfunc(_t, _rr_prev, _zz_prev, _h_off)
            if func_val_init * func_val_tmp < 0.0:
                _tmax = _t
                break #評価関数の符号がt=0.001でのものと逆符号になる最初の点で止める

                _tposmin = _t

        return _tmin, _tmax, _tnegmax, _tposmin

    def tmpfunc(self, _t, _rr_prev, _zz_prev, _h_off):
        ''' 交点判定関数（オフセット有り）(値が0の時、R_prev*Gの軌跡と指定母線が交わる) '''
        #オフセットベクトルの計算
        _d_pp, _d_qq = self.return_offvecs_dpp_dqq(_t,_h_off)
        #オフセットベクトルを考慮した試行上節点座標
        rr_tmp1, zz_tmp = self.return_next_rr_and_zz(_t,_rr_prev,_zz_prev,_d_pp,_d_qq)

        #交点判定関数の計算
        rr_tmp2 = self.gtr_rr_zz(zz_tmp)
        return rr_tmp1 - rr_tmp2

    def return_offvecs_dpp_dqq(self, _t, _h_off):
        ''' オフセットベクトルdp,dqの計算 '''

        _off_angle_pp, _off_angle_qq = self.return_offangles_pp_qq(_t)

        _d_pp = np.array([np.cos(_off_angle_pp),np.sin(_off_angle_pp)])*_h_off
        _d_qq = np.array([np.cos(_off_angle_qq),np.sin(_off_angle_qq)])*_h_off
        return _d_pp, _d_qq

    def return_offangles_pp_qq(self, _t):
        ''' オフセットベクトルdp,dqの仰角の計算 '''
        # _t > 0 (positive) : dpはbeta_pを反時計回りにpi/2回転、dqはbeta_qを時計回りにpi/2回転
        # _t < 0 (negative) : dpはbeta_pを時計回りにpi/2回転、dqはbeta_qを反時計回りにpi/2回転
        if _t > 0 :
            _off_angle_pp = self.ipl_beta_pp_t(_t) + PI/2
            _off_angle_qq = self.ipl_beta_qq_t(_t) - PI/2
        elif _t <= 0 :
            _off_angle_pp = self.ipl_beta_pp_t(_t) - PI/2
            _off_angle_qq = self.ipl_beta_qq_t(_t) + PI/2
        return _off_angle_pp, _off_angle_qq
    
    def return_next_rr_and_zz(self, _t, _rr_prev, _zz_prev, _d_pp, _d_qq):
        ''' オフセットを考慮した上点座標の計算 '''
        _rr_prev_off = _rr_prev + _d_pp[0]
        _rr_next = _rr_prev + _d_pp[0] + _rr_prev_off * self.vecgg(_t)[0] - _d_qq[0]
        _zz_next = _zz_prev + _d_pp[1] + _rr_prev_off * self.vecgg(_t)[1] - _d_qq[1]
        return _rr_next, _zz_next

    # ---------- calc 2 
    def calc_2(self):
        ''' 各層のu=0から1での挙動計算 各層でのt(u)(u=0,...,1)を格納 '''

        print("<<--- START calc_2 --->>")
        # 時間計測
        start_time = time.process_time()


#        self.list_t_u=np.array([self.num_u for i in range(self.lc)]) 
        self.list_t_u=np.zeros((self.lc,self.div_u))
        # 各層のt(u)を格納する2次元配列を初期化

        # 初層のt(u)は、初層のt0からt_limit(t0>t_Rmaxの場合), または-t_limit(t0<t_Rmaxの場合)までの等分割とする。
        self.t_limit = 0.9999999
        self.idx_u_max = 0 # uのインデックスの最大値（以下の探索により決定）

        if self.list_t0[0] > self.stdlist.t_at_rr_max :
            self.list_t_u[0] = np.linspace(self.list_t0[0], self.t_limit, self.div_u)
        else :
            self.list_t_u[0] = np.linspace(self.list_t0[0], -self.t_limit, self.div_u)
            

        if self.lc >= 2: #層数が2以上の場合
            list_t_u_tmp = np.zeros(self.lc) # 一時格納用1次元フォルダ
            for _idx_u in range(self.div_u): # u loop
#                list_t_u_tmp[0] = self.list_t_u[0][_idx_u] # 後で初期層含め格納するため、合わせて代入しておく
                list_t_u_tmp[0] = self.list_t_u[0,_idx_u] # 後で初期層含め格納するため、合わせて代入しておく
                for _idx_layer in range(1,self.lc): # layer loop
                    _h_off_prev = self.h_off #下の層のオフセット量(ここでは共通値とする)
                    _h_off_this = self.h_off #今の層のオフセット量(ここでは共通値とする)
                    _t_prev = list_t_u_tmp[_idx_layer-1] # 下の層i-1のt_i-1(u)

                    kl_prev, kl_this = self.list_kl[_idx_layer-1], self.list_kl[_idx_layer]
                    # 下層、現層のk値
                    dummy, d_qq_prev = self.return_offvecs_dpp_dqq(_t_prev,_h_off_prev)
                    # print('kl_prev,kl_this,d_qq_prev',kl_prev,kl_this,d_qq_prev)

                    # 層間接続条件
                    # 探索は、前ステップのt_i(u)±0.01とする
                    if _idx_u == 0:
                        _t_this_pre_u = self.list_t0[_idx_layer]
                    else:
#                        _t_this_pre_u = self.list_t_u[_idx_layer][_idx_u-1]
                        _t_this_pre_u = self.list_t_u[_idx_layer,_idx_u-1]

                    _t_this_min = np.max([_t_this_pre_u - 0.01, -0.9999999])
                    _t_this_max = np.min([_t_this_pre_u + 0.01, 0.9999999])

                    if _h_off_this != 0: # ヒンジオフセットの設定が有る場合
                        # t_i(u)の定義（2分法で探索。当該層のt0の値により探索方向を場合分け）
                        check = self.func_cnn(_t_this_min,_t_prev,kl_prev,kl_this,d_qq_prev,_h_off_this)*\
                            self.func_cnn(_t_this_max,_t_prev,kl_prev,kl_this,d_qq_prev,_h_off_this)
                        if check >= 0 : 
                            print("(Bisection boundary) Search stopped at _idx_u =",_idx_u)
                            flg_cp_success = False
                            break # layer loop
                        _t_this, r = opt.bisect(self.func_cnn, _t_this_min, _t_this_max, \
                              args=(_t_prev,kl_prev,kl_this,d_qq_prev,_h_off_this), full_output=True, disp=False)
                        flg_cp_success = r.converged

                    else : # ヒンジオフセットの設定が無い場合
                        _rr_this = kl_prev / kl_this * self.stdlist.ipl_ss_t(_t_prev)
                        # t_i(u)の定義 (R(t)の逆関数t(R)を使用。当該層のt0の値により_posと_negを使い分け)
                        if _rr_this < 0 or _rr_this > self.stdlist.rr_max : 
                            print("error 101")
                            flg_cp_success = False
                        elif self.list_t0[_idx_layer] > self.stdlist.t_at_rr_max :
                            _t_this = self.stdlist.ipl_t_rr_pos(_rr_this)
                            flg_cp_success = True
                        else :
                            _t_this = self.stdlist.ipl_t_rr_neg(_rr_this)
                            flg_cp_success = True

                    if flg_cp_success :
                        list_t_u_tmp[_idx_layer] = _t_this
                    else:
                        print("error 103")
                        print("Error. Ssearch stopped at _idx_u =",_idx_u)
                        break # layer loop
                    # ----- layer loop end -----
                # 全層エラー無しにt探索できた場合
                if flg_cp_success :
                    self.list_t_u[:,_idx_u] = list_t_u_tmp
                    self.idx_u_max = _idx_u
                else :
                    break # u loop
                if _idx_u % 100 == 0 : #途中経過表示
                    print(" finished idx_u = ",_idx_u)
                # ----- u loop end -----
            print(' -- result --')
            print(' idx_u_max, u_max =',self.idx_u_max, self.num_u[self.idx_u_max])
            print(' t(u_max)=',self.list_t_u[:,self.idx_u_max])

        # 終了
        end_time = time.process_time()

        print("<<--- END calc_2 --->>")

        # 処理時間出力(秒)
        proc_time = end_time - start_time
        print('calc_2 time = ',proc_time)

        return
        
    def func_cnn(self,_t_this,_t_prev,_kl_prev,_kl_this,_d_qq_prev,_h_off_this):
        _rr_this = self.stdlist.ipl_rr_t(_t_this)
        _ss_prev = self.stdlist.ipl_ss_t(_t_prev)
        _d_pp_this, dummy = self.return_offvecs_dpp_dqq(_t_this,_h_off_this)
        tmp = _kl_this*_rr_this - _d_pp_this[0] - _kl_prev*_ss_prev + _d_qq_prev[0]
        return tmp

    # ------------ calc 3
    def calc_3(self):
        ''' 各層の諸元をu=0,...,1の範囲で計算 '''

        print("<<--- START calc_3 --->>")
        # 時間計測
        start_time = time.process_time()

        if self.lc == 0 :
            print(" error! : lc = 0")  # 層が無い場合はここで終了
            return

        # 各層各ustep毎の諸元を格納
        self.list_beta_pp = self.mapping_foreach_u(self.list_t_u, self.stdlist.ipl_beta_pp_t)
        self.list_beta_qq = self.mapping_foreach_u(self.list_t_u, self.stdlist.ipl_beta_qq_t)
        self.list_phi = self.mapping_foreach_u(self.list_t_u, self.stdlist.ipl_phi_t)
        self.list_psi = self.mapping_foreach_u(self.list_t_u, self.stdlist.ipl_psi_t)
        self.list_kappa = self.mapping_foreach_u(self.list_t_u, self.stdlist.ipl_kappa_t)

        # 各層の各ustep毎の基準サイズでのR,V,S
        self.list_rr = self.mapping_foreach_u(self.list_t_u, self.stdlist.ipl_rr_t)
        self.list_vv = self.mapping_foreach_u(self.list_t_u, self.stdlist.ipl_vv_t)
        self.list_ss = self.mapping_foreach_u(self.list_t_u, self.stdlist.ipl_ss_t)

        # 各層のオフセットベクトル仰角
        self.list_offangle_pp=np.zeros((self.lc, self.div_u))
        self.list_offangle_qq=np.zeros((self.lc, self.div_u))
        for _idx_u in range(self.idx_u_max+1):
            for _idx_layer in range(self.lc):
                off_angle_pp, off_angle_qq = self.return_offangles_pp_qq(self.list_t_u[_idx_layer,_idx_u])
                self.list_offangle_pp[_idx_layer,_idx_u]=off_angle_pp
                self.list_offangle_qq[_idx_layer,_idx_u]=off_angle_qq
#        self.list_offangle_pp, self.list_offangle_qq = self.return_offangles_pp_qq(self.list_t_u)

        # 各層の下部節点(P)、上部節点(Q)の全体R,Z座標を格納

        self.list_coordr_pp=np.zeros((self.lc, self.div_u))
        self.list_coordz_pp=np.zeros((self.lc, self.div_u))
        self.list_coordr_qq=np.zeros((self.lc, self.div_u))
        self.list_coordz_qq=np.zeros((self.lc, self.div_u))
        self.list_coordr_pp_off=np.zeros((self.lc, self.div_u))
        self.list_coordz_pp_off=np.zeros((self.lc, self.div_u))
        self.list_coordr_qq_off=np.zeros((self.lc, self.div_u))
        self.list_coordz_qq_off=np.zeros((self.lc, self.div_u))

        # R,V,S,d_pp,d_qqを用いて計算。
        # ただしrr,vv,ssは基準サイズでの大きさなので、座標作成のため各層klでスケーリングする
        tmp_list_kl = self.list_kl.reshape(self.lc,1) # ブロードキャスト用に2次元配列に変換

        self.list_scaled_rr = self.list_rr * tmp_list_kl
        self.list_scaled_vv = self.list_vv * tmp_list_kl
        self.list_scaled_ss = self.list_ss * tmp_list_kl

        h_off = self.h_off
        self.list_d_pp_0 = np.cos(self.list_offangle_pp) * h_off
        self.list_d_pp_1 = np.sin(self.list_offangle_pp) * h_off
        self.list_d_qq_0 = np.cos(self.list_offangle_qq) * h_off
        self.list_d_qq_1 = np.sin(self.list_offangle_qq) * h_off

        # 各層の層間ベクトル（下部ヒンジ中央点から上部ヒンジ中央点までのベクトル）成分の計算
        vec_frame_scaled_rr = np.zeros((self.lc, self.div_u))
        vec_frame_scaled_zz = np.zeros((self.lc, self.div_u))
        for _idx_u in range(self.idx_u_max+1):
            list_t_u_tmp=self.list_t_u[:,_idx_u]
            list_d_pp_0_tmp=self.list_d_pp_0[:,_idx_u]
            list_d_pp_1_tmp=self.list_d_pp_1[:,_idx_u]
            list_d_qq_0_tmp=self.list_d_qq_0[:,_idx_u]
            list_d_qq_1_tmp=self.list_d_qq_1[:,_idx_u]

            vec_frame_scaled_rr[:,_idx_u] = list_d_pp_0_tmp + (self.list_scaled_ss[:,_idx_u]-self.list_scaled_rr[:,_idx_u])\
                 - list_d_qq_0_tmp
            vec_frame_scaled_zz[:,_idx_u] = list_d_pp_1_tmp + self.list_scaled_vv[:,_idx_u] - list_d_qq_1_tmp

        for _idx_layer in range(0, self.lc):
            # 下部ヒンジ中央点P座標の計算
            if _idx_layer == 0:
                self.list_coordz_pp[_idx_layer,:] = self.zz_min
                self.list_coordr_pp[_idx_layer,:] = self.list_scaled_rr[_idx_layer,:] - self.list_d_pp_0[_idx_layer,:]
            else :
                self.list_coordz_pp[_idx_layer,:] = self.list_coordz_pp[_idx_layer-1,:] + vec_frame_scaled_zz[_idx_layer-1,:]
                self.list_coordr_pp[_idx_layer,:] = self.list_coordr_pp[_idx_layer-1,:] + vec_frame_scaled_rr[_idx_layer-1,:]
            # 上部ヒンジ中央点Q座標の計算
            self.list_coordz_qq[_idx_layer,:] = self.list_coordz_pp[_idx_layer,:] + vec_frame_scaled_zz[_idx_layer,:]
            self.list_coordr_qq[_idx_layer,:] = self.list_coordr_pp[_idx_layer,:] + vec_frame_scaled_rr[_idx_layer,:]
            # 下部ヒンジオフセット点Poff 座標の計算
            self.list_coordz_pp_off[_idx_layer,:] = self.list_coordz_pp[_idx_layer,:] + self.list_d_pp_1[_idx_layer,:]
            self.list_coordr_pp_off[_idx_layer,:] = self.list_coordr_pp[_idx_layer,:] + self.list_d_pp_0[_idx_layer,:]
            # 上部ヒンジオフセット点Qoff 座標の計算
            self.list_coordz_qq_off[_idx_layer,:] = self.list_coordz_qq[_idx_layer,:] + self.list_d_qq_1[_idx_layer,:]
            self.list_coordr_qq_off[_idx_layer,:] = self.list_coordr_qq[_idx_layer,:] + self.list_d_qq_0[_idx_layer,:]

        # 終了
        end_time = time.process_time()

        print("<<--- END calc_3 --->>")

        # 処理時間出力(秒)
        proc_time = end_time - start_time
        print('calc_3 time = ',proc_time)

        return

    def mapping_foreach_u(self, _list_t_u, _ipl_x_t):
        ''' t_uリストとipl関数から、各層の各uに対する関数値を返す '''
        _lc, _div_u = np.shape(_list_t_u)
        return np.array([_ipl_x_t(_list_t_u[i]) for i in range(_lc)])

    def calc_elem_coord_vec(self):
        ''' 各ustepでの各層の座標軸ベクトルを計算 '''
        # 要素座標軸ベクトルの計算と格納
        # 成分は順に、層番号、uステップ番号、成分の全体座標軸番号（全体X,Y,Z）
        # 初期化
        self.list_elem_coord_x=np.zeros((self.lc, self.div_u, 3))
        self.list_elem_coord_y=np.zeros((self.lc, self.div_u, 3))
        self.list_elem_coord_z=np.zeros((self.lc, self.div_u, 3))

        for ic in range(self.lc):
            for iu in range(self.div_u):
                _t = self.list_t_u[ic,iu]
                self.list_elem_coord_x[ic,iu], self.list_elem_coord_y[ic,iu], self.list_elem_coord_z[ic,iu] \
                = self.stdlist.calc_crawford_elem_vec_t(_t)
        

  