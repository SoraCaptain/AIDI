## k8s 升级流程（脚本化）
构建新镜像并推送（tag: new-tag）。
更新 gateway-green Deployment 的镜像为新 tag。
等待绿色 Deployment 的 Pods 启动并通过 readinessProbe。
修改 Service 的 selector 从 version: blue 改为 version: green。
监控流量，确认无问题后删除蓝色 Deployment（或保留作为回滚）。

# k8s installation
Linux/WSL2:
curl -LO "https://dl.k8s.io/release/$(curl -L -s https://dl.k8s.io/release/stable.txt)/bin/linux/amd64/kubectl"
sudo install kubectl /usr/local/bin/kubectl

jetson:
curl -LO "https://dl.k8s.io/release/$(curl -L -s https://dl.k8s.io/release/stable.txt)/bin/linux/arm64/kubectl"
check version:
kubectl version --client

# Minikube
Linux:
curl -LO https://storage.googleapis.com/minikube/releases/latest/minikube-linux-amd64
sudo install minikube-linux-amd64 /usr/local/bin/minikube

jetson:
curl -LO https://storage.googleapis.com/minikube/releases/latest/minikube-linux-arm64
sudo install minikube-linux-arm64 /usr/local/bin/minikube

verfiy:
minikube version

# Start Minikube
minikube start --driver=docker --cpus=4 --memory=8192

# 查看集群节点状态
kubectl get nodes
<!-- 
NAME       STATUS   ROLES           AGE     VERSION
minikube   Ready    control-plane   6m16s   v1.35.1
-->

# 查看所有运行中的 Pod
kubectl get pods --all-namespaces

# 部署到集群
kubectl apply -f nginx-deployment.yaml

# 查看 Deployment 和 Pods
kubectl get deployments
kubectl get pods

# 部署service
kubectl apply -f nginx-service.yaml

# 查看service
kubectl get services
<!-- 
NAME            TYPE        CLUSTER-IP     EXTERNAL-IP   PORT(S)        AGE
kubernetes      ClusterIP   10.96.0.1      <none>        443/TCP        10m
nginx-service   NodePort    10.105.6.108   <none>        80:31870/TCP   13s
-->

# 访问 nginx
minikube service nginx-service

## 部署AIDI
# 创建命名空间
kubectl create namespace aidi

# 部署postgres
kubectl apply -f postgres.yaml

# 验证
kubectl get pods -n aidi

# 构建镜像
Minikube 有自己的 Docker 环境。需要先运行：
eval $(minikube docker-env)
然后再构建，这样镜像才会出现在 Minikube 的 Docker 中。

先导出环境到requirements.txt
uv export \
    --no-hashes \
    --no-emit-project \
    -o requirements.txt

docker build -f dockers/Dockerfile.gateway -t aidi-gateway:latest .