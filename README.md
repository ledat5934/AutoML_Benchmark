Yêu cầu: có thư viện google-genai<br/>
Cách chạy:<br />

- python main.py dataset_path e.g python main.py ./input/titanic (lưu ý: đường dẫn có thể khác ở trên kaggle, copy paste vào) <br />
  <br/>
  -thinking=on - bật mode thinking (nếu dùng gemini 2.5 flash)<br/>
  -m <message> - gửi thêm hướng dẫn riêng cho AI <br/>
- Trong trường hợp đường dẫn lúc sinh code khác đường dẫn runtime: <br/>
- python main.py -g <sinh_code_path> -r <runtime_path> e.g. python main.py -g input/titanic -r kaggle/input/titanic <br/>
