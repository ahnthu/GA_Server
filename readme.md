pip install flask-cors flask-restful requests numpy tensorflow fastapi uvicorn pillow opencv-python keras python-multipart
#This server is public using Ngrok
To run the server:
1. uvicorn srv:app --host 0.0.0.0 --port 8085
2. Open a new terminal and run "ngrok http 8085"
3. Use the URL that Ngrok created : "https://....-free.app" for main server