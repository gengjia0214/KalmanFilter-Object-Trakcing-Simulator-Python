import numpy as np
import cv2 as cv
from collections import deque


"""
Einsum Tricks

1 - Matrix Multiplication A.matmul(B)
np.einsum('ij,jk->ik', A, B)

2 - Batch Matrix Multiplication A.matmul(B) for A, B in A_batch, B_batch
np.einsum('bij,bjk->bik', A_batch, B_batch)

3 - Shared Matrix Multiply Batch Matrix A.matmul(B) for B in B_batch
np.eisum('ij,bjk->bik', A, B_batch)

4 - Shared Matrix Multiply Batch Matrix B.matmul(A) for B in B_batch
np.eisum('bij,jk->bik', B_batch, A)

5 - np.linalg.inv() is broadcastable!!!
for arr BxNxN
just do inv(arr) -> BxNxN
for arr NxNxB
do inv(arr.T).T  # so that the batch be the first axis when computing arr, need to transpose back
"""


class KalmanFilterMotion:
    """
    Kalman Filter for Motion Control System - works for Tracking
    Kalman filter is essentially a bayes filter with
    - linear prediction (motion) model and measurement model.
    - gaussian assumption on the process noise and measurement noise

    Graph Representation (all with right / down arrow)
    u_i is control
    x_i is state of interest
    z_i is observation/measurement
    ..    u_t-1    u_t    ..
            |       |
    .. -- x_t-1 -- x_t -- ..
            |       |
    ..    z_t-1    z_t    ..

    Bayes Recursive solution to solve
    X Posterior = measurement dist. * prediction dist. * X Posterior at t-1
    P(x_t|..) = coef * P(z_t|x_t) * Integral[P(x_t|u_t, x_t-1) * P(x_t-1|..)]dx_t-1

    Estimation
    Mean of P(x_t|..)

    Linear System with Gaussian noise
    $Motion model$
    x_t_prior = A.x_t-1_post + B.u_t + w, w ~ N(0, Q)
    P(x_t_prior) ~ P(x_t|x_t-1, u_t)P(x_t-1)
    $Measurement$
    z_t = H.x_t_prior + v, v ~ N(0, R)
    P(z_t) ~ P(z_t|x_t_prior)P(x_t_prior)

    Algorithm Essence
    - recursion
    - AN(u, s) ~ N(Au, AsA)
    - N(u1, s1)N(u2, s2) ~ N(u1 + K(u2-u1), s1 - Ks2) K is the Kalman gain K = s2/(s1 + s2)
    """

    def __init__(self, n_objects, A, B, H, Q, R, P_0_batch, X_0_batch, verbose=False):
        """
        Initialize the Kalman filter
        :param A: Transition matrix. E.g. speed & movement model 4x4
        :param B: Control matrix E.g. constant acceleration 4x4
        :param H: Encoding matrix E.g. H.X -> Z encode the state to observation (scale changing pixel -> actual)
        :param Q: Process (prediction) noise E.g. for motion control 4x4 diagonal
        :param R: Measurement noise E.g. 4x4 diagonal
        :param P_0_batch: Initial prediction error covariance E.g. 4x4 diagonal
        :param X_0_batch: Initial state. E.g. 4x1 x, y, vx, vy
        :param verbose: whether to print out the states (turn it off if you have many objects)
        """

        # pre check
        self.n_objects = n_objects
        self.size = H.shape[1]  # state shape
        assert H.shape[0] == H.shape[1]
        assert A.shape == (self.size, self.size)
        assert B.shape == (self.size, self.size)
        assert Q.shape == (self.size, self.size)
        assert R.shape == (self.size, self.size)
        assert P_0_batch.shape == (n_objects, self.size, self.size), "{}".format(P_0_batch.shape)
        assert X_0_batch.shape == (n_objects, self.size, 1), "{}".format(X_0_batch.shape)

        self.H = H
        self.A = A
        self.B = B
        self.Q = Q
        self.R = R
        P_prior = np.zeros((self.size, self.size))  # prior prediction var
        X_prior = np.zeros((self.size, 1))          # prior prediction mean

        # batch
        self.P_prior_batch = np.repeat(P_prior[np.newaxis, :], n_objects, axis=0)
        self.X_prior_batch = np.repeat(X_prior[np.newaxis, :], n_objects, axis=0)
        self.verbose = verbose
        self.P_post_batch = P_0_batch     # posterior prediction var
        self.X_post_batch = X_0_batch     # posterior prediction mean

    @staticmethod
    def get_motion_params(size):
        """
        Get the default motion model params
        :return: dict of motion model param
        """

        A = np.array([[1, 0, 1, 0],
                      [0, 1, 0, 1],
                      [0, 0, 1, 0],
                      [0, 0, 0, 1]])
        B = np.array([[0.5, 0, 0, 0],
                      [0, 0.5, 0, 0],
                      [0, 0, 1, 0],
                      [0, 0, 0, 1]])
        H = np.identity(size)

        return {'A': A, 'B': B, 'H': H}

    def filter(self, measurement: np.ndarray, control=None):
        """
        Filtering: prediction step -> correction step
        :param measurement: measurement
        :param control: motion control
        :return:
        """

        if self.verbose:
            print("\nFiltering...")

        self._prediction(control)
        self._correction(measurement)
        return self.X_post_batch, self.P_post_batch

    def _prediction(self, U_batch=None):
        """
        Prediction step.
        x_t_prior = A.x_t-1_post + B.u_t + w
        Prior prediction dist. = P(x_t|x_t-1, u_t)P(x_t-1|..)
        w ~ N(0, Q)
        P(x_t-1|..) ~ N(x_t-1_post, P_t-1_post)
        P(x_t|x_t-1, u_t)P(x_t-1|..) ~ N(A.x_t-1_post + B.u_t, AP_t-1_postA + Q)
        :param U_batch: motion control E.g. 4x1 [ax, ay, ax, ay]
        :return: void
        """

        self.A: np.ndarray
        self.B: np.ndarray
        if U_batch is None:
            U = np.zeros((self.size, 1))
            U_batch = np.repeat(U[np.newaxis, :], self.n_objects, axis=0)
        assert U_batch.shape == (self.n_objects, self.size, 1)

        # prior distribution mean
        # X_t_prior = A.x_t-1_post + B.u_t
        AX_batch = np.einsum('ij,bjk->bik', self.A, self.X_post_batch)
        BU_batch = np.einsum('ij,bjk->bik', self.B, U_batch)
        self.X_prior_batch = AX_batch + BU_batch

        # prior distribution var
        # P_t_prior = AP_t-1_postA + Q
        AP_batch = np.einsum('ij,bjk->bik', self.A, self.P_post_batch)
        APA_batch = np.einsum('bij,jk->bik', AP_batch, self.A.T)
        self.P_prior_batch = APA_batch + self.Q  # broadcast tp axis0

        if self.verbose:
            print("Prior State Estimation Mean: \n{}".format(self.X_prior_batch[0]))
            print("Prior State Estimation Var:  \n{}".format(self.P_prior_batch[0]))

    def _correction(self, Z_batch: np.ndarray):
        """
        Correction using the measurement
        z_t = H.x_t_prior + v, v ~ N(0, R)
        P(z_t|x_t) ~ N(x_prior, p_prior + R)
        P(x_t|..) ~ P(z_t|x_t)P(x_t|x_t-1, u_t)P(x_t) ~ P(z_t|x_t)P(x_t_prior)

        Product of two Gaussian
        K = P_priorH @ inv(HP_priorH+R)
        X_post = X_prior + K @ (Z - HX_prior)
        P_post = (I - KH) @ P_prior
        :param Z_batch: measurement
        :return: void
        """

        self.H: np.ndarray
        self.R: np.ndarray

        # kalman gain
        # K = P_priorH @ inv(HP_priorH+R)
        PH = np.einsum('bij,jk->bik', self.P_prior_batch, self.H.T)
        HP = np.einsum('ij,bjk->bik', self.H, self.P_prior_batch)
        HPH = np.einsum('bij,jk->bik', HP, self.H.T)    # DO NOT DO H.PH
        inv = np.linalg.inv(HPH + self.R)               # inv will broadcast along batch
        K_batch = np.einsum('bij,bjk->bik', PH, inv)   # broadcast R to HPH

        # posterior mean
        # X_post = X_prior + K @ (Z - HX_prior)
        HX = np.einsum('ij,bjk->bik', self.H, self.X_prior_batch)
        self.X_post_batch = self.X_prior_batch + np.einsum('bij,bjk->bik', K_batch, (Z_batch - HX))

        # posterior var
        # P_post = (I - KH) @ P_prior
        I = np.identity(self.size)
        KH = np.einsum('bij,jk->bik', K_batch, self.H)
        self.P_post_batch = np.einsum('bij,bjk->bik', (I - KH), self.P_prior_batch)

        if self.verbose:
            print("Posterior State Estimation Mean: \n{}".format(self.X_post_batch[0]))
            print("Posterior State Estimation Var:  \n{}".format(self.P_post_batch[0]))


class GaussianSimulator:

    def __init__(self, n_objects, p, q, r, vx=2.5, vy=2.5, ax=0.0, ay=0.0, m_var=1000, v_var=0.25, a_var=0.125,
                 acc=False):
        """
        1000x1000 Grid Simulation of objects
        :param n_objects: n objects
        :param p: initial error
        :param q: process error
        :param r: measurement error
        :param vx: init vx
        :param vy: init vy
        :param ax: init ax
        :param ay: init ay
        :param m_var: measurement var
        :param v_var: velocity var
        :param a_var: a var
        """

        # states
        self.params = [vx, vy, ax, ay]
        self.iter = 0
        self.xm = None
        self.ym = None
        self.xt = None
        self.yt = None
        self.vx = None
        self.vy = None
        self.vxm = None  # measured speed
        self.vym = None  # measured speed
        self.ax = None
        self.ay = None
        self.p = p
        self.q = q
        self.r = r
        self.acc = acc
        self.m_var = m_var
        self.v_var = v_var
        self.a_var = a_var
        self.n_objects = n_objects
        self.history = {'xm': [], 'xt': [], 'ym': [], 'yt': []}
        self.reset()

    def reset(self):

        # initialize velocity and acceleration
        vx, vy, ax, ay = self.params
        vx = np.random.normal(vx, 0.5 * vx, size=(self.n_objects, 1))
        vy = np.random.normal(vy, 0.5 * vy, size=(self.n_objects, 1))
        ax = np.random.normal(ax, 0.25 * ax, size=(self.n_objects, 1))
        ay = np.random.normal(ay, 0.25 * ay, size=(self.n_objects, 1))
        self.vx = np.random.choice([-1, 1], p=[0.5, 0.5], size=(self.n_objects, 1)) * vx
        self.vy = np.random.choice([-1, 1], p=[0.5, 0.5], size=(self.n_objects, 1)) * vy
        self.ax = np.random.choice([-1, 1], p=[0.5, 0.5], size=(self.n_objects, 1)) * ax
        self.ay = np.random.choice([-1, 1], p=[0.5, 0.5], size=(self.n_objects, 1)) * ay
        self.history = {'xm': [], 'xt': [], 'ym': [], 'yt': [], 'vxt': [], 'vyt': [], 'vxm': [], 'vym': []}
        self.iter = 0

    def step(self):
        """
        Simulate one step
        :return: measured and true position
        """

        x_noise = np.round(np.random.normal(0, self.m_var, size=(self.n_objects, 1))).astype(np.int)
        y_noise = np.round(np.random.normal(0, self.m_var, size=(self.n_objects, 1))).astype(np.int)

        # acceleration step
        self.ax = np.random.normal(self.params[2], self.a_var, size=(self.n_objects, 1))
        self.ax = self.ax * np.random.choice([-1, 1], p=[0.5, 0.5], size=(self.n_objects, 1))
        self.ay = np.random.normal(self.params[3], self.a_var, size=(self.n_objects, 1))
        self.ay = self.ay * np.random.choice([-1, 1], p=[0.5, 0.5], size=(self.n_objects, 1))
        if self.acc:
            ax, ay = self.ax, self.ay
        else:
            ax, ay = np.zeros((self.n_objects, 1)), np.zeros((self.n_objects, 1))

        # speed step
        self.vx += (ax + np.random.normal(0, self.v_var, size=(self.n_objects, 1)))
        self.vy += (ay + np.random.normal(0, self.v_var, size=(self.n_objects, 1)))

        # position step
        if self.iter == 0:
            self.xt = np.random.randint(900, 1000, size=(self.n_objects, 1), dtype=np.int)
            self.yt = np.random.randint(900, 1000, size=(self.n_objects, 1), dtype=np.int)
            self.xm, self.ym = self.xt + x_noise, self.yt + y_noise
        else:
            self.xt += np.round(self.vx + 0.5 * ax).astype(np.int)
            self.yt += np.round(self.vy + 0.5 * ay).astype(np.int)

        # measurement step
        self.xm, self.ym = self.xt + x_noise, self.yt + y_noise
        self.log()
        self.iter += 1
        # print(self.xm-self.xt, self.ym-self.yt)
        return self.xm, self.ym, self.xt, self.yt, self.vx, self.vy

    def log(self):
        self.history['xt'].append(self.xt)
        self.history['yt'].append(self.yt)
        self.history['xm'].append(self.xm)
        self.history['ym'].append(self.ym)
        self.history['vxt'].append(self.vx)
        self.history['vyt'].append(self.vy)

    def simulate(self, know_v, n_iters, filtering=True, lag=10, n_samples=5):
        """
        Show simulation
        :param know_v: whether v is known or need to be estimated by pos estimation
        :param n_iters: number of iteration
        :param filtering: whether apply filter
        :param lag: lag for measure velocity
        :param n_samples: num of samples for averaging the position for velocity measurement
        :return:
        """

        xm_queue, ym_queue = deque(), deque()
        KFfilter = None

        for i in range(n_iters):

            # get the measurement from sensor
            xm, ym, xt, yt, vxt, vyt = self.step()
            xm_queue.append(xm)
            ym_queue.append(ym)

            if know_v:
                vxm, vym = vxt, vyt
            else:  # estimate v using the measured position
                if len(xm_queue) > lag * n_samples:
                    xm_queue.popleft()
                    ym_queue.popleft()

                if len(xm_queue) > 2 * n_samples:
                    x1_sum = np.zeros_like(xm_queue[0], dtype=np.int)
                    x2_sum = np.zeros_like(x1_sum, dtype=np.int)
                    y1_sum = np.zeros_like(ym_queue[0], dtype=np.int)
                    y2_sum = np.zeros_like(y1_sum, dtype=np.int)
                    x_stack = []
                    y_stack = []

                    k = 0
                    while k < n_samples and len(xm_queue) > 0:
                        x1 = xm_queue.popleft()
                        y1 = ym_queue.popleft()
                        x_stack.append(x1)
                        y_stack.append(y1)
                        x1_sum += x1
                        y1_sum += y1
                        k += 1

                    x1 = x1_sum / 5
                    y1 = y1_sum / 5
                    while len(x_stack) > 0:
                        xm_queue.appendleft(x_stack.pop())
                        ym_queue.appendleft(y_stack.pop())

                    k = 0
                    while k < n_samples and len(xm_queue) > 0:
                        x2 = xm_queue.pop()
                        y2 = ym_queue.pop()
                        x_stack.append(x2)
                        y_stack.append(y2)
                        x2_sum += x2
                        y2_sum += y2
                        k += 1

                    x2 = x2_sum / 5
                    y2 = y2_sum / 5
                    while len(x_stack) > 0:
                        xm_queue.append(x_stack.pop())
                        ym_queue.append(y_stack.pop())

                    # x1, y1 is the averaged location of the first five measured position
                    # x2, y2 is the averaged location of the last five measured position
                    # speed measurement is debiased
                    vxm, vym = (x2 - x1) / (len(xm_queue) - n_samples//2), (y2 - y1) / (len(ym_queue) - n_samples//2)

                else:  # at the beginning, the v measurement is not stable
                    vxm, vym = (xm_queue[-1] - xm_queue[0]) / len(xm_queue), (ym_queue[-1] - ym_queue[0]) / len(ym_queue)

            # skip the first input for initialize
            if i < n_samples*lag:
                continue
            if i == n_samples*lag and filtering:
                # P0 X0 batch
                P_0 = np.diag([self.p] * 4)  # init motion model error
                X_0_batch = np.hstack((xm, ym, vxm, vym))[:, :, np.newaxis]  # Nx4x1
                P_0_batch = np.repeat(P_0[np.newaxis, :], self.n_objects, axis=0)  # Nx4x4
                # model params
                params = KalmanFilterMotion.get_motion_params(size=4)
                Q = np.diag([self.q] * 4)  # process error (motion model)
                R = np.diag([self.r] * 4)  # measurement error ()
                A, B, H = params['A'], params['B'], params['H']
                # kf instance
                KFfilter = KalmanFilterMotion(self.n_objects, A, B, H, Q, R, P_0_batch, X_0_batch, verbose=False)
            if i > n_samples*lag:
                if filtering:
                    # measurement batch tensor
                    Z_batch = np.hstack((xm, ym, vxm, vym))[:, :, np.newaxis]
                    X_post_batch, P_post_batch = KFfilter.filter(measurement=Z_batch, control=None)  # Nx4x1
                    xkf = X_post_batch[:, 0]
                    ykf = X_post_batch[:, 1]
                else:
                    xkf = np.zeros_like(xm)
                    ykf = np.zeros_like(ym)
                if self.show(xm, ym, xt, yt, xkf, ykf):
                    continue
                else:
                    break
        cv.destroyAllWindows()
        self.reset()

    def show(self, xm, ym, xt, yt, xkf, ykf):

        img = np.zeros((2048, 2048, 3), dtype=np.uint8)
        for obj_id in range(self.n_objects):
            xmi, ymi = xm[obj_id], ym[obj_id]
            xti, yti = xt[obj_id], yt[obj_id]
            xkfi, ykfi = int(xkf[obj_id][0]), int(ykf[obj_id][0])
            cv.drawMarker(img, (ymi, xmi), color=(255, 0, 0), markerType=cv.MARKER_STAR)
            cv.drawMarker(img, (yti, xti), color=(0, 255, 0), markerType=cv.MARKER_CROSS)
            cv.drawMarker(img, (ykfi, xkfi), color=(0, 0, 255), markerType=cv.MARKER_DIAMOND)
            # print(xkfi - xmi, ykfi - ymi)

        cv.imshow("im", img)
        key = cv.waitKey(60)
        if key == 27:
            return False
        return True


simulator = GaussianSimulator(1, p=25, q=0.25, r=20, vx=3, vy=3, ax=0.25, ay=0.25, m_var=30, v_var=0.3,
                              a_var=0.125, acc=False)
simulator.simulate(False, 600, filtering=True, lag=10, n_samples=5)




