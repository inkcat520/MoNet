"""
    Functions to visualize human poses
    adapted from https://github.com/una-dinosauria/human-motion-prediction/blob/master/src/viz.py
"""
import os

import imageio
import matplotlib
import matplotlib.pyplot as plt
import numpy as np

gt_color = ["#8e8e8e", "#383838"]
pre_color = ["#F48ECF", "#3535D1"]

def viz_all(dataset, subject, action, subaction, save_dir, duration=0.04):
    xyz = dataset.get_tri_xyz(dataset.data)
    plt.close('all')
    fig = plt.figure()
    ax = fig.add_subplot(projection='3d')
    ob = Ax3DPose(ax)
    plt.show(block=False)
    title_text = ax.set_title("", loc='center', pad=30.0, color="#383838")

    for sub in subject:
        for act in action:
            for subact in subaction:
                nframes = xyz[(sub, act, subact)]
                save_path = os.path.join(save_dir, f"{sub}_{act}_{subact}_all")
                print("Playing subject {0}, action {1}, subaction {2}".format(sub, act, subact))
                f_title = "subject {0}, seq:{1}_{2}".format(sub, act, subact)
                img_list = []
                for i in range(len(nframes)):
                    ob.update(nframes[i, :], nframes[i, :], pre_color)

                    # 更新标题文字
                    title_text.set_text(f_title + ' frame:{:d}'.format(i + 1))

                    # 强制重新绘制
                    fig.canvas.draw()
                    img_list.append(np.array(fig.canvas.renderer.buffer_rgba()))

                    # 等待一段时间，以便有足够的时间观察图形变化
                    plt.pause(duration)

                    # 关闭窗口
                imageio.mimsave(save_path + ".gif", img_list, duration=duration)

                plt.close()
                ax.axis('off')  # 隐藏坐标轴
                ax.set_title("", loc='center', pad=30.0, color="#383838")
                img_list = []
                for i in range(len(nframes)):
                    # 更新姿势
                    ob.update(nframes[i, :], nframes[i, :], pre_color)

                    # 强制重新绘制
                    fig.canvas.draw()
                    img_list.append(np.array(fig.canvas.renderer.buffer_rgba())[100:-100, 200:-200])

                img_list = np.hstack(img_list)
                plt.imsave(save_path + ".png", img_list)


def viz_all_cmu(dataset, subject, action, subaction, save_dir, duration=0.04):
    xyz = dataset.get_tri_xyz(dataset.data)
    plt.close('all')
    fig = plt.figure()
    ax = fig.add_subplot(projection='3d')
    ob = Ax3DPose_cmu(ax)
    plt.show(block=False)
    title_text = ax.set_title("", loc='center', pad=30.0, color="#383838")

    for sub in subject:
        for act in action:
            for subact in subaction:
                nframes = xyz[(sub, act, subact)]
                save_path = os.path.join(save_dir, f"{sub}_{act}_{subact}_all")
                print("Playing subject {0}, action {1}, subaction {2}".format(sub, act, subact))
                f_title = "subject {0}, seq:{1}_{2}".format(sub, act, subact)
                img_list = []
                for i in range(len(nframes)):
                    ob.update(nframes[i, :], nframes[i, :], pre_color)

                    # 更新标题文字
                    title_text.set_text(f_title + ' frame:{:d}'.format(i + 1))

                    # 强制重新绘制
                    fig.canvas.draw()
                    img_list.append(np.array(fig.canvas.renderer.buffer_rgba()))

                    # 等待一段时间，以便有足够的时间观察图形变化
                    plt.pause(duration)

                    # 关闭窗口
                imageio.mimsave(save_path + ".gif", img_list, duration=duration)

                plt.close()
                ax.axis('off')  # 隐藏坐标轴
                ax.set_title("", loc='center', pad=30.0, color="#383838")
                img_list = []
                for i in range(len(nframes)):
                    # 更新姿势
                    ob.update(nframes[i, :], nframes[i, :], pre_color)

                    # 强制重新绘制
                    fig.canvas.draw()
                    img_list.append(np.array(fig.canvas.renderer.buffer_rgba())[100:-100, 200:-200])

                img_list = np.hstack(img_list)
                plt.imsave(save_path + ".png", img_list)

class Ax3DPose:
    def __init__(self, ax, label=('GT', 'Pred')):
        """
        Create a 3d pose visualizer that can be updated with new poses.

        Args
          ax: 3d axis to plot the 3d pose on
          lcolor: String. Colour for the left part of the body
          rcolor: String. Colour for the right part of the body
        """

        # Start and endpoints of our representation
        self.I = np.array([1, 2, 3, 1, 7, 8, 1, 13, 14, 15, 14, 18, 19, 14, 26, 27]) - 1
        self.J = np.array([2, 3, 4, 7, 8, 9, 13, 14, 15, 16, 18, 19, 20, 26, 27, 28]) - 1
        # Left / right indicator
        self.LR = np.array([1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 0, 0, 0], dtype=bool)
        self.ax = ax

        vals = np.zeros((32, 3))

        # Make connection matrix
        self.plots = []
        for i in np.arange(len(self.I)):
            x = np.array([vals[self.I[i], 0], vals[self.J[i], 0]])
            y = np.array([vals[self.I[i], 1], vals[self.J[i], 1]])
            z = np.array([vals[self.I[i], 2], vals[self.J[i], 2]])
            if i == 0:
                self.plots.append(
                    self.ax.plot(x, z, y, lw=2, linestyle='--', c='#FFFFFF', label=label[0]))
            else:
                self.plots.append(self.ax.plot(x, y, z, lw=2, linestyle='--', c='#FFFFFF'))

        self.plots_pred = []
        for i in np.arange(len(self.I)):
            x = np.array([vals[self.I[i], 0], vals[self.J[i], 0]])
            y = np.array([vals[self.I[i], 1], vals[self.J[i], 1]])
            z = np.array([vals[self.I[i], 2], vals[self.J[i], 2]])
            if i == 0:
                self.plots_pred.append(self.ax.plot(x, y, z, lw=2, c='#FFFFFF', label=label[1]))
            else:
                self.plots_pred.append(self.ax.plot(x, y, z, lw=2, c='#FFFFFF'))

        self.ax.set_xlabel("x")
        self.ax.set_ylabel("y")
        self.ax.set_zlabel("z")
        # self.ax.set_axis_off()
        # self.ax.axes.get_xaxis().set_visible(False)
        # self.axes.get_yaxis().set_visible(False)
        self.ax.view_init(120, -90)

    def update(self, gt_channels, pred_channels, color=("#F48ECF", "#3535D1")):
        """
        Update the plotted 3d pose.

        Args
          channels: 96-dim long np array. The pose to plot.
          lcolor: String. Colour for the left part of the body.
          rcolor: String. Colour for the right part of the body.
        Returns
          Nothing. Simply updates the axis with the new pose.
        """

        assert gt_channels.size == 96, "channels should have 96 entries, it has %d instead" % gt_channels.size
        gt_vals = np.reshape(gt_channels, (32, -1))
        lcolor = "#8e8e8e"
        rcolor = "#383838"
        for i in np.arange(len(self.I)):
            x = np.array([gt_vals[self.I[i], 0], gt_vals[self.J[i], 0]])
            y = np.array([gt_vals[self.I[i], 1], gt_vals[self.J[i], 1]])
            z = np.array([gt_vals[self.I[i], 2], gt_vals[self.J[i], 2]])
            self.plots[i][0].set_xdata(x)
            self.plots[i][0].set_ydata(y)
            self.plots[i][0].set_3d_properties(z)
            self.plots[i][0].set_color(lcolor if self.LR[i] else rcolor)
            # self.plots[i][0].set_alpha(0.5)

        assert pred_channels.size == 96, "channels should have 96 entries, it has %d instead" % pred_channels.size
        pred_vals = np.reshape(pred_channels, (32, -1))
        lcolor = color[0]
        rcolor = color[1]
        for i in np.arange(len(self.I)):
            x = np.array([pred_vals[self.I[i], 0], pred_vals[self.J[i], 0]])
            y = np.array([pred_vals[self.I[i], 1], pred_vals[self.J[i], 1]])
            z = np.array([pred_vals[self.I[i], 2], pred_vals[self.J[i], 2]])
            self.plots_pred[i][0].set_xdata(x)
            self.plots_pred[i][0].set_ydata(y)
            self.plots_pred[i][0].set_3d_properties(z)
            self.plots_pred[i][0].set_color(lcolor if self.LR[i] else rcolor)
            # self.plots_pred[i][0].set_alpha(0.7)

        r = 750
        xroot, yroot, zroot = gt_vals[0, 0], gt_vals[0, 1], gt_vals[0, 2]
        self.ax.set_xlim3d([-r + xroot, r + xroot])
        self.ax.set_zlim3d([-r + zroot, r + zroot])
        self.ax.set_ylim3d([-r + yroot, r + yroot])
        self.ax.set_aspect('auto')
        self.ax.legend(loc='lower left', bbox_to_anchor=(1, 1))


def plot_predictions(expmap_gt, expmap_pred, frame_start, f_title, result_path, duration=0.05):
    # 创建一个窗口和子图
    plt.close('all')
    fig = plt.figure()
    ax = fig.add_subplot(projection='3d')

    nframes_pred = expmap_pred.shape[0]

    # 创建一个3D姿势对象
    ob = Ax3DPose(ax)

    image_list = []
    # 显示窗口，但不要阻塞程序
    plt.show(block=False)
    title_text = ax.set_title("", loc='center', pad=30.0, color="#383838")
    # 循环显示每一帧
    for i in range(nframes_pred):
        # 更新姿势
        ob.update(expmap_gt[i, :], expmap_pred[i, :], pre_color)

        # 更新标题文字
        title_text.set_text(f_title + ' frame:{:d}'.format(frame_start + i + 1))
        # ax.axis('off')  # 隐藏坐标轴
        # ax.legend().set_visible(False)  # 隐藏图例

        # 强制重新绘制
        fig.canvas.draw()
        image_list.append(np.array(fig.canvas.renderer.buffer_rgba()))

        # ax.axis('on')
        # ax.legend().set_visible(True)
        # 等待一段时间，以便有足够的时间观察图形变化
        plt.pause(duration)

    # 关闭窗口
    imageio.mimsave(result_path + ".gif", image_list, duration=duration)

    gt_list = []
    pre_list = []
    plt.close()
    ax.axis('off')  # 隐藏坐标轴
    ax.legend().set_visible(False)  # 隐藏图例
    ax.set_title("", loc='center', pad=30.0, color="#383838")
    for i in range(nframes_pred):
        # 更新姿势
        ob.update(expmap_gt[i, :], expmap_gt[i, :], gt_color)

        # 强制重新绘制
        fig.canvas.draw()
        gt_list.append(np.array(fig.canvas.renderer.buffer_rgba())[100:-100, 200:-200])

        # 更新姿势
        ob.update(expmap_gt[i, :], expmap_pred[i, :], pre_color)

        # 强制重新绘制
        fig.canvas.draw()
        pre_list.append(np.array(fig.canvas.renderer.buffer_rgba())[100:-100, 200:-200])

    pre_list = np.hstack(pre_list)
    gt_list = np.hstack(gt_list)
    plt.imsave(result_path + ".png", np.vstack((gt_list, pre_list)))


class Ax3DPose_cmu:
    def __init__(self, ax, label=('GT', 'Pred')):
        """
        Create a 3d pose visualizer that can be updated with new poses.

        Args
          ax: 3d axis to plot the 3d pose on
          lcolor: String. Colour for the left part of the body
          rcolor: String. Colour for the right part of the body
        """

        # Start and endpoints of our representation
        self.I = np.array([1, 3, 4, 1, 9, 10, 1, 15, 16, 18, 19, 16, 22, 23, 16, 31, 32]) - 1
        self.J = np.array([3, 4, 5, 9, 10, 11, 15, 16, 18, 19, 20, 22, 23, 24, 31, 32, 33]) - 1
        # Left / right indicator
        self.LR = np.array([1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 0, 0, 0], dtype=bool)
        self.ax = ax

        vals = np.zeros((38, 3))

        # Make connection matrix
        self.plots = []
        for i in np.arange(len(self.I)):
            x = np.array([vals[self.I[i], 0], vals[self.J[i], 0]])
            y = np.array([vals[self.I[i], 1], vals[self.J[i], 1]])
            z = np.array([vals[self.I[i], 2], vals[self.J[i], 2]])
            if i == 0:
                self.plots.append(
                    self.ax.plot(x, z, y, lw=2, linestyle='--', c='#FFFFFF', label=label[0]))
            else:
                self.plots.append(self.ax.plot(x, y, z, lw=2, linestyle='--', c='#FFFFFF'))

        self.plots_pred = []
        for i in np.arange(len(self.I)):
            x = np.array([vals[self.I[i], 0], vals[self.J[i], 0]])
            y = np.array([vals[self.I[i], 1], vals[self.J[i], 1]])
            z = np.array([vals[self.I[i], 2], vals[self.J[i], 2]])
            if i == 0:
                self.plots_pred.append(self.ax.plot(x, y, z, lw=2, c='#FFFFFF', label=label[1]))
            else:
                self.plots_pred.append(self.ax.plot(x, y, z, lw=2, c='#FFFFFF'))

        self.ax.set_xlabel("x")
        self.ax.set_ylabel("y")
        self.ax.set_zlabel("z")
        # self.ax.set_axis_off()
        # self.ax.axes.get_xaxis().set_visible(False)
        # self.axes.get_yaxis().set_visible(False)
        self.ax.view_init(120, -90)

    def update(self, gt_channels, pred_channels, color=("#F48ECF", "#3535D1")):
        """
        Update the plotted 3d pose.

        Args
          channels: 96-dim long np array. The pose to plot.
          lcolor: String. Colour for the left part of the body.
          rcolor: String. Colour for the right part of the body.
        Returns
          Nothing. Simply updates the axis with the new pose.
        """

        assert gt_channels.size == 114, "channels should have 114 entries, it has %d instead" % gt_channels.size
        gt_vals = np.reshape(gt_channels, (38, -1))
        lcolor = "#8e8e8e"
        rcolor = "#383838"
        for i in np.arange(len(self.I)):
            x = np.array([gt_vals[self.I[i], 0], gt_vals[self.J[i], 0]])
            y = np.array([gt_vals[self.I[i], 1], gt_vals[self.J[i], 1]])
            z = np.array([gt_vals[self.I[i], 2], gt_vals[self.J[i], 2]])
            self.plots[i][0].set_xdata(x)
            self.plots[i][0].set_ydata(y)
            self.plots[i][0].set_3d_properties(z)
            self.plots[i][0].set_color(lcolor if self.LR[i] else rcolor)
            # self.plots[i][0].set_alpha(0.5)

        assert pred_channels.size == 114, "channels should have 114 entries, it has %d instead" % pred_channels.size
        pred_vals = np.reshape(pred_channels, (38, -1))
        lcolor = color[0]
        rcolor = color[1]
        for i in np.arange(len(self.I)):
            x = np.array([pred_vals[self.I[i], 0], pred_vals[self.J[i], 0]])
            y = np.array([pred_vals[self.I[i], 1], pred_vals[self.J[i], 1]])
            z = np.array([pred_vals[self.I[i], 2], pred_vals[self.J[i], 2]])
            self.plots_pred[i][0].set_xdata(x)
            self.plots_pred[i][0].set_ydata(y)
            self.plots_pred[i][0].set_3d_properties(z)
            self.plots_pred[i][0].set_color(lcolor if self.LR[i] else rcolor)
            # self.plots_pred[i][0].set_alpha(0.7)

        r = 750
        xroot, yroot, zroot = gt_vals[0, 0], gt_vals[0, 1], gt_vals[0, 2]
        self.ax.set_xlim3d([-r + xroot, r + xroot])
        self.ax.set_zlim3d([-r + zroot, r + zroot])
        self.ax.set_ylim3d([-r + yroot, r + yroot])
        self.ax.set_aspect('auto')
        self.ax.legend(loc='lower left', bbox_to_anchor=(1, 1))


def plot_predictions_cmu(expmap_gt, expmap_pred, frame_start, f_title, result_path, duration=0.01):
    # 创建一个窗口和子图
    plt.close('all')
    fig = plt.figure()
    ax = fig.add_subplot(projection='3d')

    nframes_pred = expmap_pred.shape[0]

    # 创建一个3D姿势对象
    ob = Ax3DPose_cmu(ax)

    image_list = []
    # 显示窗口，但不要阻塞程序
    plt.show(block=False)
    title_text = ax.set_title("", loc='center', pad=30.0, color="#383838")
    # 循环显示每一帧
    for i in range(nframes_pred):
        # 更新姿势
        ob.update(expmap_gt[i, :], expmap_pred[i, :], pre_color)

        # 更新标题文字
        title_text.set_text(f_title + ' frame:{:d}'.format(frame_start + i + 1))
        # ax.axis('off')  # 隐藏坐标轴
        # ax.legend().set_visible(False)  # 隐藏图例

        # 强制重新绘制
        fig.canvas.draw()
        image_list.append(np.array(fig.canvas.renderer.buffer_rgba()))

        # ax.axis('on')
        # ax.legend().set_visible(True)
        # 等待一段时间，以便有足够的时间观察图形变化
        plt.pause(duration)

    # 关闭窗口
    imageio.mimsave(result_path + ".gif", image_list, duration=duration)

    gt_list = []
    pre_list = []
    plt.close()
    ax.axis('off')  # 隐藏坐标轴
    ax.legend().set_visible(False)  # 隐藏图例
    ax.set_title("", loc='center', pad=30.0, color="#383838")
    for i in range(nframes_pred):
        # 更新姿势
        ob.update(expmap_gt[i, :], expmap_gt[i, :], gt_color)

        # 强制重新绘制
        fig.canvas.draw()
        gt_list.append(np.array(fig.canvas.renderer.buffer_rgba())[100:-100, 200:-200])

        # 更新姿势
        ob.update(expmap_gt[i, :], expmap_pred[i, :], pre_color)

        # 强制重新绘制
        fig.canvas.draw()
        pre_list.append(np.array(fig.canvas.renderer.buffer_rgba())[100:-100, 200:-200])

    pre_list = np.hstack(pre_list)
    gt_list = np.hstack(gt_list)
    plt.imsave(result_path + ".png", np.vstack((gt_list, pre_list)))

