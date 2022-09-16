import os
import numpy as np
import cv2
import matplotlib.pyplot as plt
plt.rcParams['font.sans-serif']=['SimHei']

def calc_lines_length(lines):
    # 计算直线的长度，输入为哈夫变换函数检测出的直线数组
    # 输出每条直线长度的向量
    x1 = lines[:, :, 0]
    y1 = lines[:, :, 1]
    x2 = lines[:, :, 2]
    y2 = lines[:, :, 3]
    d_x = x1 - x2
    d_y = y1 - y2
    d = np.sqrt(d_x * d_x + d_y * d_y)
    return d


def calc_line_equ_ABC(x1, y1, x2, y2):
    # 计算直线的一般方程Ax + By + C = 0
    # 输入直线的两点坐标，可为向量
    # 输出直线的一般方程系数(A, B, C)。
    # 注1：如果输入为向量，则计算每一个直线方程并输出向量
    # 注2：此函数会使计算出的B始终大于等于0
    A = y2 - y1
    B = x1 - x2
    C = (x2 - x1) * y1 - (y2 - y1) * x1
    A, B, C = np.atleast_1d(A, B, C)
    negative = B < 0
    A[negative] *= -1
    B[negative] *= -1
    C[negative] *= -1
    return (A, B, C)


def calc_distance_signed(line_ABC, p):
    # 计算点到直线的距离，但不会取绝对值，返回可能为负
    x0 = p[0]
    y0 = p[1]
    A, B, C = line_ABC
    return (A * x0 + B * y0 + C) / np.sqrt(A * A + B * B)


def calc_line_equ_normal(x1, y1=None, x2=None, y2=None):
    # 计算直线的法线式方程，返回(ρ, α)
    # 输入直线两点坐标或直线一般方程的系数ABC
    # 注：如果输入为向量，则计算每一个方程并返回向量
    if y1 is None:
        ABC = x1
        theta = calc_theta(ABC)
    else:
        theta = calc_theta(x1, y1, x2, y2)
        ABC = calc_line_equ_ABC(x1, y1, x2, y2)
    alpha = np.pi / 2 + theta
    rho = -calc_distance_signed(ABC, (0, 0))
    return (rho, alpha)


def remove_inner_lines(lines, max_dalpha):
    # 删除内部直线函数，输入为哈夫变换函数检测出的直线数组、近平行直线的最大角度差，输出为相同形式的直线数组
    # 利用此函数可有效去除由于封面图案而检测出的过长的内部直线
    # 具体做法为：
    #      1 设置一个向量flag，来标记每条直线是否要保留，刚开始flag向量元素全为False
    #      2 遍历每条直线，分别执行如下操作：
    #         2.1 找出该条直线的所有近平行直线，近平行直线是指与该直线夹角小于输入参数max_dalpha的直线，组成近平行直线组
    #         2.2 找出近平行直线组中相距最远的两条直线，标记设为True
    # 注：找出近平行直线组中最远两条直线方法为：找出法线式方程中ρ最大和ρ最小的两条直线
    x1 = lines[:, :, 0].flatten()
    y1 = lines[:, :, 1].flatten()
    x2 = lines[:, :, 2].flatten()
    y2 = lines[:, :, 3].flatten()
    theta = calc_theta(x1, y1, x2, y2)
    flag = np.zeros(lines.shape[0], dtype=np.bool8)
    rho, alpha = calc_line_equ_normal(x1, y1, x2, y2)
    # print(alpha.shape)
    alpha2D_row = alpha.reshape((1, -1))
    dalpha = line_theta_diff(np.dot(alpha2D_row.T, np.ones_like(alpha2D_row)),
                             np.dot(np.ones_like(alpha2D_row).T, alpha2D_row))
    # print(dalpha.shape)
    # show_lines(lines, im, (0, 0, 255))
    for i in range(dalpha.shape[0]):
        row_dalpha = dalpha[i, :]
        pal_ind = np.where(row_dalpha < max_dalpha)[0]
        rho_temp = rho.copy()
        if np.minimum(alpha[i], np.pi - alpha[i]) < max_dalpha:
            rho_temp[alpha > np.pi / 2] *= -1

        # ind = np.argsort(rho[pal_ind])
        # show_lines(lines[pal_ind], im)
        max_ind = np.argmax(rho_temp[pal_ind])
        min_ind = np.argmin(rho_temp[pal_ind])
        flag[pal_ind[min_ind]] = True
        flag[pal_ind[max_ind]] = True
        # print(rho[pal_ind])
        # show_lines(lines[flag], im, (0, 255, 255))

    return lines[flag]


def remove_outer_lines(lines, im_shape, padding):
    # 去除外线函数
    # 功能为：删去图片边缘的直线
    # lines为哈夫变换函数检测出的直线数组，im_shape为图像大小，padding为阈值，离图像边缘近于padding的直线会被删除
    # 返回相同形式的直线数组
    x1 = lines[:, :, 0].flatten()
    y1 = lines[:, :, 1].flatten()
    x2 = lines[:, :, 2].flatten()
    y2 = lines[:, :, 3].flatten()
    w = im_shape[1]
    h = im_shape[0]
    # valid = np.ones(lines.shape[0], dtype=np.bool8)
    left_most = padding
    right_most = w - padding - 1
    top_most = padding
    bottom_most = h - padding - 1
    valid = (x1 >= left_most) * (x1 <= right_most) * (x2 >= left_most) * (x2 <= right_most) * \
            (y1 >= top_most) * (y1 <= bottom_most) * (y2 >= top_most) * (y2 <= bottom_most)
    return lines[valid]


def clean_lines(lines, thresh, max_count):
    # 清除杂乱直线函数
    # 输入哈夫变换函数检测出的直线数组、距离阈值、最大聚集数，输出相同形式直线数组
    # 做法：统计每条直线两端点附近（thresh指定范围）其它直线（端点）的个数，如果个数超过max_count，则删去该直线
    lines = lines.copy()
    count = np.ones(lines.shape[0], dtype=np.int32)
    x1 = lines[:, :, 0]
    y1 = lines[:, :, 1]
    x2 = lines[:, :, 2]
    y2 = lines[:, :, 3]
    for i in range(len(lines)):
        for j in range(i + 1, len(lines)):
            x11 = x1[i, 0]
            x12 = x2[i, 0]
            x21 = x1[j, 0]
            x22 = x2[j, 0]
            y11 = y1[i, 0]
            y12 = y2[i, 0]
            y21 = y1[j, 0]
            y22 = y2[j, 0]
            d1 = calc_distance(x11, y11, x21, y21)
            d2 = calc_distance(x11, y11, x22, y22)
            d3 = calc_distance(x12, y12, x21, y21)
            d4 = calc_distance(x12, y12, x22, y22)
            d = min([d1, d2, d3, d4])
            if d < thresh:
                count[i] += 1
                count[j] += 1
                # break
    slice = count <= max_count
    return lines[slice]


def line_theta_diff(theta1, theta2):
    # 计算两直线的夹角，输入为两直线的倾斜角度
    dtheta = np.fabs(theta1 - theta2)
    return np.minimum(dtheta, np.pi - dtheta)


def calc_theta(x1, y1=None, x2=None, y2=None):
    # 计算直线的倾斜角度，输入可为直线两点坐标，也可为一般方程参数A，B，C
    # 如果输入为向量，则分别计算每条直线，返回一向量
    with np.errstate(divide='ignore'):
        if y1 is None:
            A, B, C = x1
            return np.arctan(-np.double(B) / A)
        else:
            return np.arctan((np.double(y2) - y1) / (x2 - x1))


def calc_distance(x1, y1, x2, y2):
    # 计算两点距离
    # 如果输入为向量，则分别计算，返回一向量
    dx = x1 - x2
    dy = y1 - y2
    return np.sqrt(dx * dx + dy * dy)


def show_lines(lines, im, rgb=(255, 0, 0), text=None, line_thickness=2, show_immediately=True):
    # 将直线标记在图像上并显示，可以指定颜色、文字等
    # 若指定文字，则对应参数位置应为一列表，长度与lines相同，表示每条直线上要显示的文字
    im_show = im.copy()
    for i in range(lines.shape[0]):
        x1 = lines[i][0][0]
        y1 = lines[i][0][1]
        x2 = lines[i][0][2]
        y2 = lines[i][0][3]
        cv2.line(im_show, (x1, y1), (x2, y2), rgb, line_thickness)
        cv2.circle(im_show, (x1, y1), 7, (0, 0, 0), 4)
        cv2.circle(im_show, (x2, y2), 7, (0, 0, 0), 4)
        if not text is None:
            cv2.putText(im_show, str(text[i]), (x1 + 7, y1 - 7), fontFace=cv2.FONT_HERSHEY_DUPLEX, fontScale=1,
                        color=rgb)
    # print(lines.shape)
    plt.imshow(im_show)
    if show_immediately:
        plt.show()


def connect_lines(lines, thresh1=np.pi / 90, thresh2=np.pi / 125):
    # 直线合并函数
    # 功能：将直线数组中相同的两条直线合并为一条，合并后的直线两端点为合并前两直线所有四个端点中相距最远的两个
    # 方法：
    # 直线之间两两比较，如果两直线倾斜角度相近（夹角小于thresh1），则认为两直线平行，再进行以下判断：
    #       尝试从两直线中各取一点，连成一条新直线，
    #       如果这条直线与原直线倾斜角度相近（夹角小于thresh2），则认为原两直线是同一条直线，合并
    #       注：当一条直线与多条直线平行时，会选取连成的新直线中与原直线倾斜角度最近似的直线进行合并
    # 合并的方法为：取两直线所有四个端点中相距最远的两点，作为新直线
    lines_slice = np.ones(lines.shape[0], dtype=np.bool8)
    lines = lines.copy()
    x1 = lines[:, :, 0]
    y1 = lines[:, :, 1]
    x2 = lines[:, :, 2]
    y2 = lines[:, :, 3]
    rho, alpha = calc_line_equ_normal(x1, y1, x2, y2)
    rho = rho.flatten()
    alpha = alpha.flatten()
    theta = calc_theta(x1, y1, x2, y2).flatten()
    alpha_row = alpha.reshape((1, -1))
    for i in range(len(lines)):
        if i + 1 >= len(lines):
            break
        alpha_i = alpha[i]
        rho_i = rho[i]
        x1_i = x1[i, 0]
        y1_i = y1[i, 0]
        x2_i = x2[i, 0]
        y2_i = y2[i, 0]
        dalpha = line_theta_diff(alpha_i, alpha[i + 1:])
        d_rho = np.fabs(rho[i + 1:] - rho_i)
        parallel = dalpha < thresh1
        if np.any(parallel):
            pal_ind = np.nonzero(parallel)[0]
            theta_attempt1 = calc_theta(x1_i, y1_i, x1[i + 1:].flatten(), y1[i + 1:].flatten()).flatten()[pal_ind]
            theta_diff11 = line_theta_diff(theta_attempt1, theta[i])
            theta_diff12 = line_theta_diff(theta_attempt1, theta[i + 1:][pal_ind])
            theta_diff1 = np.maximum(theta_diff11, theta_diff12)
            theta_attempt2 = calc_theta(x1_i, y1_i, x2[i + 1:].flatten(), y2[i + 1:].flatten()).flatten()[pal_ind]
            theta_diff21 = line_theta_diff(theta_attempt2, theta[i])
            theta_diff22 = line_theta_diff(theta_attempt2, theta[i + 1:][pal_ind])
            theta_diff2 = np.maximum(theta_diff21, theta_diff22)
            theta_diff = np.minimum(theta_diff1, theta_diff2)
            min_ind = np.argmin(theta_diff)
            if theta_diff[min_ind] > thresh2:
                continue

            cnt_ind = pal_ind[min_ind] + i + 1
            lines_slice[cnt_ind] = True
            lines_slice[i] = False
            x11 = x1[i, 0]
            x12 = x2[i, 0]
            x21 = x1[cnt_ind, 0]
            x22 = x2[cnt_ind, 0]
            y11 = y1[i, 0]
            y12 = y2[i, 0]
            y21 = y1[cnt_ind, 0]
            y22 = y2[cnt_ind, 0]
            if np.fabs(theta[i]) < np.pi / 4:
                p1_index = np.argmin([x11, x12, x21, x22])
                p2_index = np.argmax([x11, x12, x21, x22])
                x1_new = [x11, x12, x21, x22][p1_index]
                x2_new = [x11, x12, x21, x22][p2_index]
                y1_new = [y11, y12, y21, y22][p1_index]
                y2_new = [y11, y12, y21, y22][p2_index]
            else:
                p1_index = np.argmin([y11, y12, y21, y22])
                p2_index = np.argmax([y11, y12, y21, y22])
                x1_new = [x11, x12, x21, x22][p1_index]
                x2_new = [x11, x12, x21, x22][p2_index]
                y1_new = [y11, y12, y21, y22][p1_index]
                y2_new = [y11, y12, y21, y22][p2_index]

            # show_lines(lines[[i, cnt_ind]], im, rgb=(0, 0, 255), text=[[x11, y11, x12, y12], [x21, y21, x22, y22]])
            # print([[x11, y11, x12, y12], [x21, y21, x22, y22]])
            lines[cnt_ind, 0, :] = [x1_new, y1_new, x2_new, y2_new]
            # show_lines(lines[lines_slice], im)
    return lines[lines_slice]


def calc_node(ABC1, ABC2):
    # 计算两直线交点
    A1, B1, C1 = ABC1
    A2, B2, C2 = ABC2
    x = (B1 * C2 - B2 * C1) / (A1 * B2 - A2 * B1)
    y = (A1 * C2 - A2 * C1) / (A2 * B1 - A1 * B2)
    return (x, y)


def calc_vector_angle(v1, v2):
    # 计算向量夹角
    return np.arccos(np.inner(v1, v2) / (np.linalg.norm(v1, axis=-1) * np.linalg.norm(v2, axis=-1)))


def calc_pts_img(lines, start_vector=np.array([-1, 0])):
    # 计算透视图形在图像中的4个顶点坐标
    # 具体功能：根据围成透视图形的四条直线计算出对应的四个顶点坐标，按顺时针方向排列四个顶点并返回
    # 返回顺序：以四个顶点的均值中心为原点放置向量start_vector，将start_vector沿顺时针旋转一周依次碰到的四个点
    # 方法：
    #     1 取出其中一条直线，计算与其余三条直线的夹角，与其夹角最小的另一条直线认为是该直线的对边
    #     2 将该直线与其对边分别计算与其余两条直线的交点，获得4个交点坐标
    #     3 计算4个交点的均值中心点，从该中心点出发与4个交点分别构成4个向量
    #     4 计算4个向量与向量start_vector的夹角
    #      （注：此处夹角范围为0-2*pi，非传统定义，定义如下：
    #       将一向量沿顺时针旋转，如果在某一位置与另一向量同向，则与该向量夹角为原向量转过的角度）
    #     5 按与向量start_vector夹角将4个交点从小到大排列，输出
    x1 = lines[:, :, 0].flatten()
    y1 = lines[:, :, 1].flatten()
    x2 = lines[:, :, 2].flatten()
    y2 = lines[:, :, 3].flatten()
    ABC = calc_line_equ_ABC(x1, y1, x2, y2)
    rho, alpha = calc_line_equ_normal(ABC)
    A, B, C = ABC
    ABC = np.vstack([A, B, C])
    # print(ABC)
    dalpha = line_theta_diff(alpha[1:], alpha[0])
    relative_line_ind = np.argmin(dalpha) + 1
    slice_ABC1 = np.zeros(shape=(4), dtype=np.bool8)
    slice_ABC1[[0, relative_line_ind]] = True
    ABC1 = ABC[:, slice_ABC1]
    ABC2 = ABC[:, np.logical_not(slice_ABC1)]
    ABC11 = ABC1[:, [0]]
    ABC12 = ABC1[:, [1]]
    ABC21 = ABC2[:, [0]]
    ABC22 = ABC2[:, [1]]
    x, y = calc_node(np.hstack([ABC11, ABC11, ABC12, ABC12]), np.hstack([ABC21, ABC22, ABC22, ABC21]))
    pts = np.hstack([x.reshape(-1, 1), y.reshape(-1, 1)])
    pts_mean = pts.mean(axis=0)
    pts_vector = pts - pts_mean
    angle = calc_vector_angle(pts_vector, start_vector)
    z = np.cross(start_vector, pts_vector)
    angle[z < 0] = np.pi * 2 - angle[z < 0]
    ind = np.argsort(angle)
    pts_img = np.around(pts[ind]).astype('float32')
    return pts_img


def estimate_pts_obj(pts_img, diag_size):
    # 估算透视图形原图顶点坐标，参数为透视图形在图像中的4个顶点坐标、原图的对角线长度（以像素为单位）
    # 方法：
    #     1 计算图像中的2对相对边各自的平均长度，得出原图长宽比例
    #     2 根据原图对角线长度diag_size计算4个顶点坐标，组成数组返回
    pts_img1 = np.roll(pts_img, -1, axis=0)
    x1 = pts_img[:, 0]
    y1 = pts_img[:, 1]
    x2 = pts_img1[:, 0]
    y2 = pts_img1[:, 1]
    d = calc_distance(x1, y1, x2, y2)
    # k^2 * w^2 + k^2 * h^2 = diag_size^2
    w = d[0] + d[2]
    h = d[1] + d[3]
    k = diag_size / np.sqrt(w * w + h * h)
    w *= k
    h *= k
    w = np.round(w)
    h = np.round(h)
    pts_obj = np.array([[0, 0], [w, 0], [w, h], [0, h]], dtype=np.float32)
    return pts_obj


def calc_pts_obj(wh, diag_size):
    # 计算原图形顶点坐标，wh = (w, h)指定原图形长宽比，diag_size指定原图形的对角线长度（以像素为单位）
    w = wh[0]
    h = wh[1]
    k = diag_size / np.sqrt(w*w + h*h)
    w *= k
    h *= k
    w = np.round(w)
    h = np.round(h)
    pts_obj = np.array([[0,0],[w,0],[w,h],[0,h]], dtype=np.float32)
    return pts_obj


def rectify_doc(img, w_h=None, start_vector=None, show_every_step=False):
    # ***文档矫正函数***
    # 输入：原图像；输出：校正后的图像

    if len(img.shape) > 2 and img.shape[2] != 1:
        gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    else:
        gray = img
    if start_vector is None:
        start_vector = (0, 1)

    # 图像预处理（目前无预处理）
    im = img
    gray_b = gray

    # 直线拟合（Canny边缘检测、哈夫变换拟合直线）
    edges = cv2.Canny(gray_b, threshold1=40, threshold2=100)
    lines = cv2.HoughLinesP(edges, 1, np.pi / 180, 35, minLineLength=30, maxLineGap=10)
    if show_every_step:
        plt.subplot(231)
        plt.axis('off')
        plt.title('1.直线拟合')
        show_lines(lines, im, line_thickness=3, show_immediately=False)

    # 清除杂乱直线
    lines = clean_lines(lines, 15, 5)
    if show_every_step:
        plt.subplot(232)
        plt.axis('off')
        plt.title('2.清除杂乱直线')
        show_lines(lines, im, line_thickness=3, show_immediately=False)

    # 合并直线
    lines_cnt = connect_lines(lines)
    if show_every_step:
        plt.subplot(233)
        plt.axis('off')
        plt.title('3.合并直线')
        show_lines(lines_cnt, im, line_thickness=3, show_immediately=False)

    # 清除外线（图像边缘线）
    lines_cnt = remove_outer_lines(lines_cnt, im.shape, 10)
    if show_every_step:
        plt.subplot(234)
        plt.axis('off')
        plt.title('4.清除外线')
        show_lines(lines_cnt, im, line_thickness=3, show_immediately=False)

    # 清除内线（文档内部线）
    d = calc_lines_length(lines_cnt).flatten()
    lines_cnt = remove_inner_lines(lines_cnt[d > d.max() * 0.3], np.pi / 11)
    if show_every_step:
        plt.subplot(235)
        plt.axis('off')
        plt.title('5.清除内线')
        show_lines(lines_cnt, im, line_thickness=3, show_immediately=False)

    # 找出最长的4条线
    d = calc_lines_length(lines_cnt).flatten()
    d_ind = np.argsort(d, axis=0)
    lines_max4 = lines_cnt[d_ind[-4:]]
    if show_every_step:
        plt.subplot(236)
        plt.axis('off')
        plt.title('6.找出最长的4条线')
        show_lines(lines_max4, im, line_thickness=3, show_immediately=False)
        plt.pause(1)

    # 计算图像中顶点坐标、估计原图形顶点坐标
    pts_img = calc_pts_img(lines_max4, start_vector=start_vector)
    if w_h is None:
        pts_obj = estimate_pts_obj(pts_img, diag_size=1400)
    else:
        pts_obj = calc_pts_obj(w_h, diag_size=1400)
    obj_width, obj_height = int(pts_obj[2, 0]), int(pts_obj[2, 1])

    # 文档矫正
    M = cv2.getPerspectiveTransform(pts_img, pts_obj)
    warped = cv2.warpPerspective(im, M, (obj_width, obj_height))

    if show_every_step:
        plt.figure(2)
        plt.axis('off')
        plt.title('矫正后图形')
        # plt.subplot(236)
        plt.imshow(warped)
        plt.show()
    return warped


if __name__ == "__main__":
    # (图像路径, 宽高, 顶点排列顺序向量(见函数calc_pts_img介绍))
    img_list = [
        ('doc1.jpg', (17, 24), (0, 1)),
        ('doc2.jpg', (11.2, 18.2), (0, 1)),
        ('doc3.jpg', (14, 20), (-1, 0)),
        ('bad_example.jpg', (11, 14), (-1, 0))
        ]

    img_info = img_list[0]   # ***修改此处索引以得到各图片矫正结果***

    img_path, img_wh, start_vector = img_info
    img_path = os.path.join('imgs', img_path)
    img = cv2.imread(img_path)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    warped = rectify_doc(img, w_h = img_wh, start_vector=start_vector, show_every_step=True)

    save_path = os.path.join('imgs', img_path[:-4] + '_warped.jpg')
    cv2.imwrite(img_path[:-4] + '_warped.jpg', cv2.cvtColor(warped, cv2.COLOR_RGB2BGR))
