# 睿聘智模 v3.1 · AI 智能面试系统

> 部署路径示例：`D:\interview_agent\web_agent`

---

## 📁 项目文件结构

```
web_agent/
├── app.py               # Flask 后端（主程序）
├── config.py            # 配置文件（从 .env 读取敏感参数）
├── interviewer_engine.py# 核心面试引擎
├── index.html           # 前端页面（与 app.py 同级）
├── requirements.txt     # Python 依赖列表
├── .env.example         # 环境变量模板（需复制为 .env 并填写）
├── .env                 # 【你需要创建此文件】存放 API Key 等敏感配置
├── start.bat            # Windows 一键启动脚本
├── start.sh             # Mac/Linux 一键启动脚本
└── static/              # 静态资源目录（可为空，Flask 使用）
```

---

## 🚀 快速部署（Windows）

### 第一步：安装 Python

- 下载 Python 3.10 或更高版本：https://www.python.org/downloads/
- 安装时勾选 **"Add Python to PATH"**

### 第二步：创建 .env 配置文件

在 `D:\interview_agent\web_agent\` 目录下执行：

```bat
copy .env.example .env
```

用记事本打开 `.env`，填写您的 API Key：

```
API_KEY=sk-你的真实APIKey
MODEL_NAME=gpt-4o
```

### 第三步：启动项目

**方式一（推荐）：** 双击 `start.bat`

**方式二（命令行）：**
```bat
cd /d D:\interview_agent\web_agent
pip install -r requirements.txt
python app.py
```

### 第四步：访问系统

打开浏览器访问：**http://localhost:5000**

---

## 🛠️ 常见问题排查

| 错误信息 | 原因 | 解决方法 |
|---------|------|---------|
| `ModuleNotFoundError: No module named 'dotenv'` | 缺少 python-dotenv | `pip install python-dotenv` |
| `[睿聘智模] 未找到 API_KEY` | .env 文件不存在或未填写 | 检查 `.env` 文件是否存在且 `API_KEY` 已填写 |
| `Address already in use` | 5000 端口被占用 | 在 `.env` 中修改 `SERVER_PORT=8080` |
| 浏览器打开显示 404 | index.html 路径问题 | 确认 `index.html` 与 `app.py` 在同一目录 |
| PDF/Word 解析失败 | 缺少解析库 | `pip install PyMuPDF python-docx` |

---

## ⚙️ .env 配置说明

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `API_KEY` | 无（必填）| OpenAI API Key |
| `API_BASE` | `https://api.openai.com/v1` | API 地址，中转服务可修改 |
| `MODEL_NAME` | `gpt-4o` | 主对话模型 |
| `SERVER_PORT` | `5000` | 服务端口 |
| `LLM_TIMEOUT` | `120` | API 超时秒数 |

---

## 📦 依赖版本要求

- Python >= 3.10
- openai >= 1.0.0
- flask >= 3.0.0
- python-dotenv >= 1.0.0（**必须安装**）
- PyMuPDF >= 1.23.0
- python-docx >= 1.1.0
- reportlab >= 4.0.0