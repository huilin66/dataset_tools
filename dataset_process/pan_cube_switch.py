

def p2c(panorama_path, cube_size):
    cube_dice = np.array(Image.open(panorama_path))
    cubemap = py360convert.e2c(cube_dice, face_w=cube_size, cube_format='list')
    return cubemap


def cube2pcd(cube, coor_xy, cube_poss, cube_size):
    def get_poss(input_box):
        x_min, y_min, x_max, y_max = input_box
        lt = [x_min, y_min]
        rt = [x_max, y_min]
        lb = [x_min, y_max]
        rb = [x_max, y_max]
        mt = [int(0.5 * (x_min + x_max)), y_min]
        mb = [int(0.5 * (x_min + x_max)), y_max]
        lm = [x_min, int(0.5 * (y_min + y_max))]
        rm = [x_max, int(0.5 * (y_min + y_max))]
        poss = [lt, rt, lb, rb, mt, mb, lm, rm]
        return poss

    dst_pos = []
    for cube_pos in cube_poss:
        poss = get_poss(cube_pos)
        poss_tf = []
        for pos in poss:
            pos_x, pos_y = pos
            if cube == 'up':
                pos_x_tf = pos_x + cube_size * 4
                pos_y_tf = pos_y
            elif cube == 'forward' or cube == 'front':
                pos_x_tf = pos_x + cube_size * 0
                pos_y_tf = pos_y
            elif cube == 'right':
                pos_x_tf = pos_x + cube_size * 1
                pos_y_tf = pos_y
            elif cube == 'back':
                pos_x_tf = pos_x + cube_size * 2
                pos_y_tf = pos_y
            elif cube == 'left':
                pos_x_tf = pos_x + cube_size * 3
                pos_y_tf = pos_y
            elif cube == 'down':
                pos_x_tf = pos_x + cube_size * 5
                pos_y_tf = pos_y
            else:
                pass
            pos_y_tf = int(pos_y_tf + 0.5)
            pos_x_tf = int(pos_x_tf + 0.5)
            if pos_y_tf >= cube_size:
                pos_y_tf = cube_size - 1
            if pos_x_tf >= cube_size * 6:
                pos_x_tf = cube_size * 6 - 1
            poss_tf.append(coor_xy[pos_y_tf][pos_x_tf])
        dst_pos.append(poss_tf)
    return np.array(dst_pos)


def c2p():
    xyz = py360convert.utils.xyzcube(cube_size)
    uv = py360convert.utils.xyz2uv(xyz)
    coor_xy = py360convert.utils.uv2coor(uv, cube_size * 2, cube_size * 4)
    pan_poss = cube2pcd(cube=cube_name, coor_xy=coor_xy, cube_poss=cube_bboxs[:, 2:], cube_size=cube_size)