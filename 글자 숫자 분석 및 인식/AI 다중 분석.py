import cv2
import numpy as np
import tensorflow as tf

# ------- 1. 레이블 정의 -------
mnist_labels = [str(i) for i in range(10)]                  # 0~9
emnist_labels = [chr(i) for i in range(ord('A'), ord('Z')+1)]  # A~Z
all_labels = mnist_labels + emnist_labels                   # 총 36개

# ------- 2. 모델 불러오기 -------
model = tf.keras.models.load_model("emnist_mnist_combined_model.h5")

# ------- 3. 그림판 설정 -------
canvas = np.zeros((280, 280), dtype=np.uint8)
drawing = False
window_name = "🖌️ 마우스로 숫자 또는 알파벳을 그리세요 (C: 초기화 / Enter: 완료)"

def draw(event, x, y, flags, param):
    global drawing
    if event == cv2.EVENT_LBUTTONDOWN:
        drawing = True
    elif event == cv2.EVENT_MOUSEMOVE and drawing:
        cv2.circle(canvas, (x, y), 10, 255, -1)
    elif event == cv2.EVENT_LBUTTONUP:
        drawing = False

cv2.namedWindow(window_name)
cv2.setMouseCallback(window_name, draw)

# ------- 4. 그리기 루프 -------
while True:
    cv2.imshow(window_name, canvas)
    key = cv2.waitKey(1) & 0xFF
    if key == ord('c'):
        canvas[:] = 0
    elif key == 13:  # Enter
        break

cv2.destroyAllWindows()

# ------- 5. 이미지 전처리 -------
img = cv2.resize(canvas, (28, 28))
img = 1 - (img / 255.0)  # 흰색 배경일 경우 반전
img = img.reshape(1, 28, 28, 1)

# ------- 6. 예측 -------
prediction = model.predict(img)
pred_index = np.argmax(prediction)
pred_label = all_labels[pred_index]

# ------- 7. 숫자/영문 자동 판별 -------
if pred_label in mnist_labels:
    category = "숫자"
else:
    category = "영문자"

# ------- 8. 결과 출력 -------
print("\n🎯 AI 인식 결과")
print(f" - 분류: {category}")
print(f" - 문자: {pred_label}")
