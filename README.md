# Phần mềm báo bia tự động bằng Computer Vision

# "computer vision là gì?"
Khái niệm: Computer Vision (Thị giác máy tính) là một lĩnh vực trong trí tuệ nhân tạo (AI) và khoa học máy tính, tập trung vào việc phát triển các hệ thống và thuật toán giúp máy tính "nhìn thấy" và hiểu được thế giới xung quanh thông qua hình ảnh và video. Mục tiêu của computer vision là để máy tính có thể nhận diện, phân tích, và hiểu được các đối tượng, cảnh vật, và các đặc trưng trong hình ảnh hoặc video giống như cách con người nhìn và hiểu thế giới.

# cách phần mềm hoạt động:
- chụp lại hình ảnh bia loạt trước, và sau khi bắn, lưu lại để kiểm tra
- sau đó so sánh khác biệt giữa các loạt để tìm ra lỗ đạn mới trên bia, đánh dấu lỗ đạn và ghi số loạt bắn lên lỗ đạn để phân biệt giữa loạt trước và loạt sau
- xác định các thông số của các vòng tròn/elip đồng tâm để tính toán điểm của từng phát đạn
- áp dụng công thức elip để xác định nếu điểm nằm trong elip
- đã tối ưu thuật toán để giảm thiểu ảnh hưởng của điều kiện ánh sáng, rung lắc bia, rung lắc camera

# cách sử dụng phần mềm: 
1. Mở phần mềm lên
2. Gắn camera vào
3. Bấm nút "Thêm Camera" để máy tính quét và nhận diện các camera đã kết nối
4. Nhập thông tin của camera (VD: Bia gì, dải số mấy)
5. Bấm nút "Thêm Dải bắn", nếu có 10 dải thì thêm đúng đến dải thứ 10, và thêm đầy đủ camera cho mỗi bia/mỗi dải (VD bài 1 AK: 3 mục tiêu, 10 dải bắn thì thêm đủ 30 camera)

6. Bấm nút "Bắt đầu bắn" để máy tính chụp và lưu lại hình ảnh tất cả các bia "an toàn" để đối chiếu với loạt bắn thứ nhất
7. Hô khẩu lệnh "nằm/quỳ/đứng chuẩn bị bắn"
8. Hô khẩu lệnh "mục tiêu hiện" và các loạt đồng thời bắn
9. Khi các dải bắn xong, hô khẩu lệnh ngưng bắn khám súng
10. Hô khẩu lệnh "Báo bia" và đồng thời bấm nút "Báo bia" để máy tính tính toán các lỗ đạn đã phát hiện và báo điểm tự động (ví dụ loạt 1 bệ số 1 trúng 2 viên: 9 9 = 18 điểm)
11. Người chỉ huy sẽ nhìn vào kết quả máy tính trả về để báo điểm cho các bệ bắn (VD: loạt 1 bệ số 1: 18 điểm, đạt yêu cầu)
kết quả này sẽ được lưu lại trong thư mục "C:/HinhAnh/KetQua/DaiBan1/"

12. Hô khẩu lệnh đổi tập, và loạt tiếp theo sẽ lên tuyến bắn (loạt 2)
13. Bấm nút "Bắt đầu bắn loạt tiếp theo" để máy tính chuẩn bị tạo ra thêm 1 loạt bắn
14. Hô các khẩu lệnh điều hành bắn như thường cho đến khi loạt 2 bắn xong hô ngưng bắn khám súng
15. Hô khẩu lệnh "Báo bia" và đồng thời bấm nút "Báo bia" để máy tính báo kết quả
16. tương tự bước 11

17. các loạt tiếp theo làm tương tự loạt 2, kể từ loạt 2 không được bấm lại nút "bắt đầu bắn" nữa

# CÁCH TIẾP TỤC PHÁT TRIỂN PHẦN MỀM TRONG TƯƠNG LAI
1. các file:
- main.py: file chính dùng để gọi các module, các hàm để chạy phần mềm
- camera.py: gồm các class, hàm để làm việc với camera
- image_processing.py: gồm các hàm để nhận diện và xử lý hình ảnh, phân biệt lỗ đạn, báo bia
- object_detection.py: gồm các hàm để nhận diện vật thể trong hình

# 2. các hàm:
# I. main.py:
1. update_variables()
Hàm này cập nhật các giá trị tham số toàn cục từ các thanh trượt (slider) và hiển thị các giá trị đã thay đổi trong giao diện người dùng.
Các bước hoạt động:
Lấy giá trị từ các thanh trượt tương ứng với các tham số như giá trị làm mờ, ngưỡng thích ứng, ngưỡng nhị phân, v.v.
Cập nhật thông tin này vào nhãn result_label để hiển thị các giá trị đã thay đổi.

2. open_variable_editor()
Hàm này mở một cửa sổ con mới để người dùng có thể chỉnh sửa các tham số toàn cục (như giá trị làm mờ, ngưỡng, v.v.).
Các bước hoạt động:
Tạo cửa sổ mới (Toplevel).
Thêm các thanh trượt (slider) để người dùng có thể chỉnh sửa các giá trị tham số.
Khi người dùng bấm nút "Áp dụng", hàm update_variables() sẽ được gọi để cập nhật các tham số.

3. detect_cameras_thread()
Hàm này được dùng để phát hiện các camera USB có kết nối với máy tính. Hàm này sẽ kiểm tra liên tục các chỉ số camera và nếu phát hiện có camera mới, nó sẽ mở cửa sổ chỉnh sửa camera cho người dùng.
Chú thích: Đoạn mã này hiện chưa được triển khai đầy đủ (do phần kiểm tra camera bị chú thích).

4. create_camera_object(camera_id)
Hàm này sẽ tạo một đối tượng camera mới dựa trên thông tin người dùng nhập vào và lưu nó vào danh sách camera_objects.
Các bước hoạt động:
Lấy thông tin về mục tiêu (target) và lane (dải bắn) từ các trường nhập liệu.
Kiểm tra tính hợp lệ của thông tin và tạo đối tượng camera tương ứng.

5. open_variable_editor(camera_id)
Hàm này mở cửa sổ để người dùng có thể nhập thông tin cho camera (như dải bắn và mục tiêu).
Các bước hoạt động:
Tạo cửa sổ mới (Toplevel) cho phép người dùng nhập thông tin camera.
Sau khi nhập thông tin, nhấn nút "Áp dụng và Đóng" sẽ gọi hàm create_camera_object(camera_id) để tạo camera.

6. check_camera_and_open_editor()
Hàm này kiểm tra xem các camera có kết nối hay không. Nếu phát hiện camera, nó sẽ mở cửa sổ chỉnh sửa cho camera đó.
Các bước hoạt động:
Duyệt qua các chỉ số camera và kiểm tra xem camera có được phát hiện không.
Nếu camera mới được phát hiện, hàm sẽ mở cửa sổ chỉnh sửa camera.

7. show_result()
Hàm này sẽ hiển thị kết quả chụp ảnh từ các làn bắn và các vòng bắn.
Các bước hoạt động:
Duyệt qua tất cả các làn và vòng bắn.
Tải và hiển thị các ảnh kết quả từ thư mục tương ứng.

8. start_shooting()
Hàm này bắt đầu quá trình chụp ảnh cho các bia mục tiêu. Nó tạo các thư mục con cho từng làn bắn và sau đó gọi hàm chụp ảnh.
Các bước hoạt động:
Tạo các thư mục chứa ảnh cho mỗi làn bắn.
Bắt đầu quá trình chụp ảnh bằng cách gọi phương thức parallel_capture().

9. add_shooting_lane()
Hàm này thêm một dải bắn mới vào giao diện người dùng (thêm một tab vào Notebook).
Các bước hoạt động:
Tăng số lượng dải bắn (num_lane).
Thêm một tab mới cho dải bắn vào notebook.

10. add_shooting_turn()
Hàm này thêm một loạt bắn mới.
Các bước hoạt động:
Tăng số loạt bắn (num_turn).
Cập nhật thông báo số loạt bắn đã thêm.

11. review_result(img, lane, turn, target)
Hàm này sẽ xử lý và xem xét lại kết quả ảnh sau khi bắn (chưa hoàn thiện, chỉ có cấu trúc hàm).

12. shooting_turn_complete()
Hàm này hoàn tất một vòng bắn và chụp ảnh kết quả.
Các bước hoạt động:
Gọi lại hàm parallel_capture() để chụp ảnh cho vòng bắn hiện tại.
Sử dụng compare_and_detect() để so sánh và phát hiện kết quả bia.

13. reset()
Hàm này sẽ reset lại số lượng làn bắn và vòng bắn, đồng thời xóa tất cả các tab trong notebook.
Các bước hoạt động:
Đặt lại giá trị của num_lane và num_turn về 0.
Loại bỏ tất cả các tab trong giao diện và hiển thị thông báo.

14. get_current_tab()
Hàm này trả về tab hiện tại đang được chọn trong Notebook.

15. on_submit(param1, param2, param3, window)
Hàm này được gọi khi người dùng bấm nút "Thêm" trong cửa sổ thêm camera. Nó nhận các tham số như dải bắn, mục tiêu và camera ID, sau đó gọi add_camera().

16. add_camera(lane, target, camera_id)
Hàm này thêm một camera vào danh sách và hiển thị thông báo.

17. add_camera_form()
Hàm này tạo một cửa sổ con mới để người dùng nhập thông tin camera như dải bắn, mục tiêu và camera ID.
Các bước hoạt động:
Tạo cửa sổ mới để nhập thông tin camera.
Gửi thông tin khi người dùng nhấn nút "Thêm".

Các nút trong giao diện:
Bắt đầu bắn: Gọi hàm start_shooting() để bắt đầu chụp ảnh cho bia.
Thêm Dải Bắn: Gọi hàm add_shooting_lane() để thêm một dải bắn mới.
Bắt đầu bắn loạt tiếp theo: Gọi hàm add_shooting_turn() để thêm một vòng bắn mới.
Báo bia: Gọi hàm shooting_turn_complete() để hoàn thành vòng bắn và chụp ảnh.
Chỉnh sửa tham số: Gọi hàm open_variable_editor() để mở cửa sổ chỉnh sửa tham số.
Thêm Camera: Gọi hàm add_camera_form() để thêm một camera mới.

# II. image_processing.py:
# 1. load_image(lane, turn, target)
Mô tả: Hàm này tải một ảnh từ đường dẫn dựa trên thông tin về dải bắn, loạt bắn, và tên bia.
Tham số:
lane: Dải bắn.
turn: Loạt bắn.
target: Tên bia.
Trả về: Trả về ảnh đã tải.

# 2. load_result(lane, turn, target)
Mô tả: Hàm này tải kết quả ảnh đã được đánh dấu từ đường dẫn.
Tham số:
lane: Dải bắn.
turn: Loạt bắn.
target: Tên bia.
Trả về: Trả về ảnh kết quả đã đánh dấu.

# 3. is_hole_inside_ellipse(x, y, h, k, a, b, angle)
Mô tả: Kiểm tra xem một điểm có nằm trong một elip hay không, dựa vào các tham số như tọa độ điểm và các tham số của elip.
Tham số:
x, y: Tọa độ của điểm cần kiểm tra.
h, k: Tọa độ tâm của elip.
a, b: Bán trục lớn và bán trục nhỏ của elip.
angle: Góc quay của elip.
Trả về: True nếu điểm nằm trong elip, ngược lại trả về False.

# 4. get_bullet_holes(lane, turn)
Mô tả: Trả về các lỗ đạn của một dải bắn và loạt bắn cụ thể.
Tham số:
lane: Dải bắn.
turn: Loạt bắn.
Trả về: Các lỗ đạn của dải bắn và loạt bắn tương ứng.

# 5. get_elipse_target_center(image)
Mô tả: Tìm vị trí của tên bia có hình elip trong ảnh, bao gồm các tham số như bán trục, tâm và góc quay của elip.
Tham số:
image: Ảnh chứa tên bia.
Trả về: Các tham số của elip gồm bán trục lớn, bán trục nhỏ, tọa độ tâm và góc quay.

# 6. draw_debug_elipse(image, a, b, h, k, angle)
Mô tả: Vẽ elip trên ảnh để kiểm tra và debug.
Tham số:
image: Ảnh cần vẽ elip lên.
a, b: Bán trục của elip.
h, k: Tọa độ tâm của elip.
angle: Góc quay của elip.

# 7. calculate_score(lane, turn, target)
Mô tả: Tính toán điểm số cho các lỗ đạn dựa trên vị trí của chúng so với elip.
Tham số:
lane: Dải bắn.
turn: Loạt bắn.
target: Tên bia.
Trả về: Không có giá trị trả về, nhưng thông báo điểm số sẽ được hiển thị.

# 8. on_image_click(event, canvas, img, text_entries, lane, turn, target)
Mô tả: Hàm này xử lý sự kiện khi người dùng nhấp chuột lên ảnh để đánh dấu lỗ đạn và tính toán điểm.
Tham số:
event: Sự kiện chuột nhấp.
canvas: Kênh vẽ (canvas) trên giao diện người dùng.
img: Ảnh cần xử lý.
text_entries: Các ô nhập liệu.
lane: Dải bắn.
turn: Loạt bắn.
target: Tên bia.

# 9. save_image(image, lane, turn, target)
Mô tả: Lưu ảnh đã xử lý vào thư mục với tên động.
Tham số:
image: Ảnh cần lưu.
lane: Dải bắn.
turn: Loạt bắn.
target: Tên bia.

# 10. is_hole_already_exist(x, y, w, h)
Mô tả: Kiểm tra xem lỗ đạn đã tồn tại hay chưa, nếu có thì không thêm vào.
Tham số:
x, y: Tọa độ của lỗ đạn.
w, h: Kích thước của lỗ đạn.
Trả về: True nếu lỗ đạn đã tồn tại, False nếu chưa.

# 11. draw_debug(image, x, y, r, turn)
Mô tả: Vẽ hộp bao quanh lỗ đạn và đánh dấu loạt bắn lên ảnh.
Tham số:
image: Ảnh cần vẽ.
x, y: Tọa độ của lỗ đạn.
r: Bán kính của lỗ đạn.
turn: Loạt bắn.

# 12. circularity_check(image, x, y, r)
Mô tả: Kiểm tra tính tròn của lỗ đạn bằng cách tính circularity và độ tương phản.
Tham số:
image: Ảnh cần kiểm tra.
x, y: Tọa độ trung tâm của vòng tròn.
r: Bán kính của vòng tròn.
Trả về: True nếu vòng tròn có circularity hợp lệ, False nếu không.

# 13. compare_and_detect(lane, turn, target)
Mô tả: So sánh hai ảnh (trước và sau) của một loạt bắn để phát hiện sự thay đổi và các lỗ đạn.
Tham số:
lane: Dải bắn.
turn: Loạt bắn.
target: Tên bia.

# III. camera.py:




