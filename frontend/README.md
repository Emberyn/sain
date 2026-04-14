# 前端文件结构说明

## 文件拆分完成

为了提高代码的可维护性和可读性，已将原来的 `index.html` 拆分为三个独立文件：

### 📁 文件结构

```
frontend/
├── index.html      (8.2KB)  - HTML结构文件
├── style.css       (11.1KB) - CSS样式文件
├── script.js       (50.2KB) - JavaScript逻辑文件
└── index_old.html  (78.6KB) - 原始未拆分的备份文件
```

### 📝 各文件职责

#### 1. index.html
- 包含HTML页面结构
- 引用外部CSS和JS文件
- 保持简洁的HTML标记

#### 2. style.css
- 所有CSS样式定义
- 包括响应式设计、动画、主题颜色等
- 使用CSS变量便于主题定制

#### 3. script.js
- 所有JavaScript业务逻辑
- 包括图片上传、掩码处理、Canvas绘图等
- API调用和结果展示

### 🔧 后端配置

`app.py` 已正确配置静态文件服务：

```python
app = Flask(__name__, static_folder='frontend', static_url_path='/')
```

Flask会自动处理以下请求：
- `/` → 返回 `index.html`
- `/style.css` → 返回 `style.css`
- `/script.js` → 返回 `script.js`

### ✅ 优势

1. **代码组织清晰**：HTML、CSS、JS分离，易于维护
2. **浏览器缓存**：CSS和JS文件可以被浏览器缓存，提高加载速度
3. **团队协作**：不同开发者可以并行编辑不同文件
4. **代码复用**：样式和脚本可以在多个页面间共享
5. **调试友好**：浏览器开发者工具可以更清晰地显示错误位置

### 🗑️ 清理建议

如果确认新文件工作正常，可以删除备份文件：
```bash
rm frontend/index_old.html
```

### 🚀 使用方法

启动服务器后访问：
```
http://localhost:5000
```

所有功能与之前完全相同，只是代码组织更加合理。
