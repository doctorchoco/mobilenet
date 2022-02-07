from val import *
import torchvision.transforms as T
from torchvision.datasets import ImageFolder
import os
from torch.utils.data.dataset import Dataset
from glob import glob


class InferDataset(Dataset):
    def __init__(self, labels, images_folder):
        super().__init__()
        with open(labels, 'r') as f:
            self._labels = json.load(f)
        self._images_folder = images_folder
    def __getitem__(self, idx):
        file_name = self._labels['images'][idx]['file_name']
        img = cv2.imread(os.path.join(self._images_folder, file_name), cv2.IMREAD_COLOR)
        return {
            'img': img,
            'file_name': file_name
        }
    def __len__(self):
        return len(self._labels['images'])


def image_crop(img, save_path, net, multiscale=False, visualize=False, crop=True, save=False):
    base_height = 368
    scales = [1]
    if multiscale:
        scales = [0.5, 1.0, 1.5, 2.0]
    stride = 8

    # 1. 입력 이미지가 작을 경우 종료
    if min(img.shape[0], img.shape[1]) < 128: return

    avg_heatmaps, avg_pafs = infer(net, img, scales, base_height, stride)

    total_keypoints_num = 0
    all_keypoints_by_type = []

    for kpt_idx in range(18):  # 19th for bg
        total_keypoints_num += extract_keypoints(avg_heatmaps[:, :, kpt_idx], all_keypoints_by_type, total_keypoints_num)

    pose_entries, all_keypoints = group_keypoints(all_keypoints_by_type, avg_pafs)

    # 2. 검출된 Pose가 없을 경우 종료
    if len(pose_entries) == 0:
        print("None")
        return

    coco_keypoints, scores = convert_to_coco_format(pose_entries, all_keypoints)

    # 3. Score 가 일정 Threshold 이하일 경우 종료
    if (scores[0] < 100):
        return

    cropped_image = None

    if crop:
        for keypoints in coco_keypoints:
            y = []
            x = []
            z = []
            for idx in range(len(keypoints) // 3):
                if visualize:
                    cv2.circle(img, (int(keypoints[idx * 3]), int(keypoints[idx * 3 + 1])),
                               3, (255, 0, 255), -1)
                x.append(int(keypoints[idx * 3]))
                y.append(int(keypoints[idx * 3 + 1]))
                z.append(int(keypoints[idx * 3 + 2]))

            """
            keypoints index
            0 코 / 1 오른눈 / 2 왼눈 / 3 오른귀 / 4 왼귀
            5 오른어깨 / 6 왼어깨 / 7 오른팔꿈치 / 8 왼팔꿈치 / 9 오른팔목 / 10 왼팔목
            11 오른골반 / 12 왼골반
            13 오른무릎 / 14 왼무릎
            15 오른발 / 16 왼발
            """

            # key points of human face
            face_indices = [0, 1, 2]
            cnt = 0
            for idx in face_indices:
                if z[idx]==1: cnt += 1
            # Front 여부 판단
            if cnt < 3: continue

            # key points of human body
            body_indices = [5, 6, 7, 8, 11, 12]
            nx = []
            ny = []
            nz = []
            for idx in body_indices:
                if z[idx]==1:
                    nx.append(x[idx])
                    ny.append(y[idx])
                    nz.append(z[idx])

            # 우 key point가 3개씩 없는 경우
            if sum(nz[1::2]) < 3: continue
            # 좌 key point가 3개씩 없는 경우
            if sum(nz[::2]) < 3: continue
            # 좌우 key point가 cross 된 경우
            croos_cnt = 0
            for i in range(0,len(nx),2):
                if nx[i] < nx[i+1]:
                    croos_cnt += 1
            if croos_cnt > 0:
                continue

            ## Find min-max and draw bounding-box
            x_min, x_max = min(nx), max(nx)
            xl = int((x_max - x_min)*0.1)
            y_min, y_max = min(ny), max(ny)
            yl = int((y_max - y_min)*0.2)

            xi = max(x_min-xl, 0)
            xe = min(x_max+xl, img.shape[0])
            yi = max(y_min-yl, 0)
            ye = min(y_max+yl, img.shape[1])
            cropped_image = img[yi:ye,xi:xe,:]

        if save:
            save_folder = save_path + file_name.split("/")[-2]
            if not os.path.exists(save_folder):
                os.mkdir(save_folder)
            cv2.imwrite(save_path + "/".join(file_name.split("/")[-2:]), cropped_image)

        if visualize:
            cv2.rectangle(img, (xi, yi), (xe, ye), (255, 0, 0), 2)
            plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
            plt.savefig("figures/" + file_name.split("/")[-1], dpi=300)
            plt.show()

    return cropped_image


def load_pose_model(model_path, device):
    net = PoseEstimationWithMobileNet()
    checkpoint = torch.load(model_path, map_location=device)
    load_state(net, checkpoint)
    net = net.eval()
    return net


if __name__=="__main__":
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
#    data_path = "/input2/share/dataset/dale_sample_dataset/dataset_1018/trainset/1/*"
#    data_path = "/input2/share/dataset/coco/val/*"
#    data_path = "/input2/share/dataset/dale_raw/20210201/*"
    data_paths = "/input2/share/dataset/Action_dataset/action_small/*/*.jpg"
    save_path = "/input2/share/dataset/Action_dataset/action_small_crop/"
    net = load_pose_model(model_path = "models/checkpoint_iter_370000.pth", device=device)
    datasets = glob(data_paths)

    for file_name in datasets:
        try:
            img = cv2.imread(os.path.join(file_name), cv2.IMREAD_COLOR)
            cropped_image = image_crop(img, save_path, net, multiscale=False, visualize=False, save=True)
#            print(coco_keypoints, scores)
        except Exception as e:
            print("Error")


    """

        image_id = int(file_name[0:file_name.rfind('.')])
        for idx in range(len(coco_keypoints)):
            coco_result.append({
                'image_id': image_id,
                'category_id': 1,  # person
                'keypoints': coco_keypoints[idx],
                'score': scores[idx]
            })

        if visualize:
            for keypoints in coco_keypoints:
                for idx in range(len(keypoints) // 3):
                    cv2.circle(img, (int(keypoints[idx * 3]), int(keypoints[idx * 3 + 1])),
                               3, (255, 0, 255), -1)
            cv2.imshow('keypoints', img)
            key = cv2.waitKey()
            if key == 27:  # esc
                return

    with open(output_name, 'w') as f:
        json.dump(coco_result, f, indent=4)

    run_coco_eval(labels, output_name)
    
    """