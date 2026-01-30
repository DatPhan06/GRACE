# Backend Architecture

backend viết bằng fastapi, dùng db là postgresql, được tổ chức theo kiến trúc phân lớp (Layered Architecture) với 4 folder chính, đi từ thấp đến cao:

## 1. Infra (`/infra`)
- **Vai trò**: Layer thấp nhất.
- **Nhiệm vụ**: Chứa các service dùng để kết nối với bên ngoài, hệ thống bên thứ 3, hoặc database.
- **Ví dụ**: Database connection (SQLAlchemy, asyncpg), Redis client, Email service client, S3 client.

## 2. Domain (`/domain`)
- **Vai trò**: Chứa logic nghiệp vụ cốt lõi.
- **Nhiệm vụ**: Định nghĩa các entities, models, và các logic xử lý cụ thể cho từng domain. Độc lập với framework web (FastAPI) càng nhiều càng tốt.
- **Ví dụ**: User entity, Order logic, Pricing rules.

## 3. App (`/app`)
- **Vai trò**: Orchestration layer (Logic ứng dụng).
- **Nhiệm vụ**: Kết hợp các domain để hoàn thành một use case cụ thể của ứng dụng.
- **Ví dụ**: `CreateUserUseCase` (gọi domain User để tạo, gọi infra Email để gửi mail chào mừng), `ProcessOrderUseCase`.

## 4. Api (`/api`)
- **Vai trò**: Layer cao nhất (Interface layer).
- **Nhiệm vụ**: Expose các tính năng ra bên ngoài qua giao thức HTTP (FastAPI). Nhận request, validate input, gọi xuống layer App/Domain, và trả về response.
- **Ví dụ**: FastAPI routers, Pydantic schemas (request/response models), Dependency injection setup.

## Quản lý dự án
- **Monorepo**: Build theo hướng monorepo.
- **Package Manager**: Sử dụng `uv` để quản lý dependencies và môi trường.
