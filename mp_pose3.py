import cv2
import mediapipe as mp
import numpy as np
import pygame
import time

mp_drawing = mp.solutions.drawing_utils
mp_pose = mp.solutions.pose

# 檢查點是否在圓內的函數
def is_point_inside_circle(point, circle_center, circle_radius):
    return np.linalg.norm(np.array(point) - np.array(circle_center)) <= circle_radius

# 初始化 Pygame 以播放聲音
pygame.mixer.init()
hit_sound = pygame.mixer.Sound("Pop_up_sound.wav")

# 對於靜態圖像：
IMAGE_FILES = []
with mp_pose.Pose(
    static_image_mode=True,
    model_complexity=2,
    min_detection_confidence=0.5) as pose:
    
    # 創建列表以存儲下落的球
    balls = []

    for idx, file in enumerate(IMAGE_FILES):
        image = cv2.imread(file)
        image_height, image_width, _ = image.shape
        results = pose.process(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))

        if not results.pose_landmarks:
            continue

        # 檢查是否有任何姿勢標誌在圓內
        for landmark in results.pose_landmarks.landmark:
            landmark_point = (int(landmark.x * image_width), int(landmark.y * image_height))
            for ball in balls:
                if is_point_inside_circle(landmark_point, ball['center'], ball['radius']):
                    ball['click_count'] += 1
                    ball['color'] = (0,0,255)
                    if ball['click_count'] >= 3:
                        score += 1
                        balls.remove(ball)
                        hit_sound.play()  # 播放命中音效

        # 更新下落球的位置
        for ball in balls:
            ball['center'] = (ball['center'][0], ball['center'][1] + 5)  # 調整下落速度

        # 有一定概率添加新球
        if np.random.rand() < 0.02: # 根據需要調整概率
            new_ball = {
                'center': (np.random.randint(50, 500), 0),
                'radius': 30,
                'color': [(255, 0, 0)],
                'opacity': 1.0,
                'click_count': 0
            }
            balls.append(new_ball)

        # 在圖像上繪製球
        overlay = image.copy()
        for ball in balls:
            cv2.circle(overlay, ball['center'], ball['radius'], ball['color'], -1)  # -1 fills the circle
        cv2.addWeighted(overlay, 1.0, image, 0.5, 0, image)  # Adjust the alpha for transparency

        # 繪製姿勢全球標誌
        annotated_image = image.copy()
        mp_drawing.draw_landmarks(
            annotated_image, results.pose_landmarks, mp_pose.POSE_CONNECTIONS)
        cv2.imwrite('/tmp/annotated_image' + str(idx) + '.png', annotated_image)


        # Plot pose world landmarks.
        mp_drawing.plot_landmarks(
            results.pose_world_landmarks, mp_pose.POSE_CONNECTIONS)

# 對於攝像頭輸入：
cap = cv2.VideoCapture(0)
with mp_pose.Pose(
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5) as pose:
    
     # 創建列表以存儲下落的球
    balls = []
    score = 0  # 初始化得分
    start_time = time.time()
    countdown_duration = 30  # 設置倒計時持續時間（秒）
    timer_font = cv2.FONT_HERSHEY_SIMPLEX
    timer_font_size = 1
    timer_font_color = (0, 0, 0)
    timer_font_thickness = 2

    while cap.isOpened():
        success, image = cap.read()
        if not success:
            print("Ignoring empty camera frame.")
            continue
        image = cv2.cvtColor(cv2.flip(image, 1), cv2.COLOR_BGR2RGB)
        image.flags.writeable = False
        results = pose.process(image)

        remaining_time = max(countdown_duration - int(time.time() - start_time), 0)
        timer_text = f'Time: {remaining_time} s'
        image_umat = cv2.UMat(image)
        cv2.putText(image_umat, timer_text, (10, 70), timer_font, timer_font_size, timer_font_color, timer_font_thickness)

        # 檢查是否有任何姿勢標誌在圓內
        if results.pose_landmarks == None:
            pass
        else:
            for landmark in results.pose_landmarks.landmark:
                landmark_point = (int(landmark.x * image.shape[1]), int(landmark.y * image.shape[0]))
                for ball in balls:
                    if is_point_inside_circle(landmark_point, ball['center'], ball['radius']):
                        ball['click_count'] += 1
                        ball['color'] = (0,0,255)
                        if ball['click_count'] >= 3:                        
                            # 如果點擊3次，則移除球
                            balls.remove(ball)
                            hit_sound.play()
                            score += 1

        # 更新下落球的位置
        for ball in balls:
            ball['center'] = (ball['center'][0], ball['center'][1] + 5)  # 調整下落速度

        # 有一定概率添加新球
        if np.random.rand() < 0.02: 
            new_ball = {
                'center': (np.random.randint(50, 550), 0),
                'radius': 30,
                'color': (255, 0, 0),
                'opacity': 1.0,
                'click_count': 0
            }
            balls.append(new_ball)

        # 在圖像上繪製球
        overlay = image.copy()
        for ball in balls:
            cv2.circle(overlay, ball['center'], ball['radius'], ball['color'], -1)  # -1填充圓
        cv2.addWeighted(overlay, 1.0, image_umat, 0.5, 0, image_umat)  # 調整alpha以控制透明度

        # 在圖像上繪製姿勢標誌
        # 將UMat對象轉換為numpy陣列，因為使用了cv2.putText和cv2.imshow函數，這些函數可能無法直接處理UMat對象。
        image_np = cv2.UMat.get(image_umat)
        image.flags.writeable = True
        image_np = cv2.cvtColor(image_np, cv2.COLOR_RGB2BGR)
        # print(image_umat.get().shape)
        # print(image_umat.get())
        mp_drawing.draw_landmarks(
            image_np, results.pose_landmarks, mp_pose.POSE_CONNECTIONS)
        cv2.putText(image_np, f'Score: {score}', (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 0), 2)
        cv2.imshow('MediaPipe Pose', image_np)
        
        k = cv2.waitKey(5)

        if time.time() - start_time >= 30:
            print(f'Total Score: {score}')
            break
            
        if k == ord("q"):
            break

cap.release()
cv2.destroyAllWindows()
