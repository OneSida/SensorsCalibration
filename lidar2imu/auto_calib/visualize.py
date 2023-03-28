import numpy as np
import ujson
import open3d as o3d


def format_pose_file(input_file, output_file):
    fin = open(input_file)
    fout = open(output_file, 'w')
    for line in fin:
        line = ' '.join(line.split(' ')[1:])
        fout.write(line)
    fin.close()
    fout.close()


def load_extrinsic(input_file):
    fin = open(input_file)
    obj = ujson.load(fin)
    fin.close()
    return obj['gnss-to-top_center_lidar-extrinsic']['param']['sensor_calib']['data']


def transform_lidar_to_imu(extrinsic_file, input_file, output_file):
    final_T = np.array(load_extrinsic(extrinsic_file))

    fin = open(input_file)
    fout = open(output_file, 'w')
    for line in fin:
        line = line.strip().split(' ')
        lidar_pose = np.zeros((4, 4))
        lidar_pose[3, 3] = 1.
        for i, item in enumerate(line):
            row = int(i / 4)
            col = i % 4
            lidar_pose[row, col] = float(item)
        transformed_lidar_pose = np.dot(np.linalg.inv(final_T), np.dot(lidar_pose, final_T))

        out_list = []
        for i in range(12):
            row = int(i / 4)
            col = i % 4
            out_list.append('{:.6f}'.format(transformed_lidar_pose[row, col]))
        fout.write('{}\n'.format(' '.join(out_list)))
    fin.close()
    fout.close()


def plot_pcd(extrinsic_file, ins_file, output_pcd_file, downsample=200):
    final_T = np.array(load_extrinsic(extrinsic_file))

    fin = open(ins_file)
    global_homo_points = []
    for line in fin:
        line = line.strip().split(' ')
        local_pcd_file = './data/top_center_lidar/{}.pcd'.format(line[0])
        pcd = o3d.io.read_point_cloud(local_pcd_file)
        points = np.array(pcd.points)[::downsample, ...]
        local_homo_points = np.concatenate((points.T, np.ones((1, points.shape[0]))))

        imu_pose = np.zeros((4, 4))
        imu_pose[3, 3] = 1.
        for i, item in enumerate(line[1:]):
            row = int(i / 4)
            col = i % 4
            imu_pose[row, col] = float(item)
        lidar_pose = np.dot(final_T, np.dot(imu_pose, np.linalg.inv(final_T)))

        global_homo_points.append(np.dot(lidar_pose, local_homo_points))
    global_homo_points = np.concatenate(global_homo_points, axis=1)
    fusion_pcd = o3d.geometry.PointCloud()
    fusion_pcd.points = o3d.utility.Vector3dVector(global_homo_points[:3, :].T)
    o3d.io.write_point_cloud(output_pcd_file, fusion_pcd)
    # o3d.visualization.draw_geometries([fusion_pcd])

    fin.close()


if __name__ == '__main__':
    # format_pose_file('./data/NovAtel-pose-lidar-time.txt', './imu_pose.txt')
    # format_pose_file('./data/top_center_lidar-pose.txt', './lidar_pose.txt')
    # transform_lidar_to_imu('./data/gnss-to-top_center_lidar-extrinsic.json', \
    #   'lidar_pose.txt', 'transformed_lidar_pose.txt')
    plot_pcd('./data/gnss-to-top_center_lidar-extrinsic.json', \
        './data/NovAtel-pose-lidar-time.txt', 'fusion_init.pcd')
    plot_pcd('./data/result.gnss-to-top_center_lidar-extrinsic.json', \
        './data/NovAtel-pose-lidar-time.txt', 'fusion_result.pcd')
    pass

