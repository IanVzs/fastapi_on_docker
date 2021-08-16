# fastapi_on_docker
fastapi 运行在docker的最小示例

## 构建image
```bash
docker build -t myimage .
docker run -d --name fastapi_on_docker -p 8080:80 myimage
```
