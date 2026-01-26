**models**
helmet_detection - https://drive.google.com/file/d/1mtSTXbBtJlVZ0gA0p0tvZl2PcASaptJ_/view?usp=sharing
head_detection  -  https://drive.google.com/file/d/1-QwwZh7zrdHEhe9Abzs1YnUzg_I9T3x3/view?usp=sharing


1.Build Image
```bash
docker build -t helmet-detection .
```
2.Run the Container (RTSP + Web)
```bash
docker run --rm \
  --network host \
  -v $(pwd)/models:/app/models \
  -v $(pwd)/captures:/app/captures \
  helmet-detection
```
3. View Live Visualization
```bash
http://localhost:5000/video
```
