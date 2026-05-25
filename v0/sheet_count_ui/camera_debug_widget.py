from PySide6.QtWidgets import QFormLayout, QHBoxLayout, QLabel, QLineEdit, QPushButton, QVBoxLayout, QWidget


class CameraDebugWidget(QWidget):
    def __init__(self, main_window):
        super().__init__(main_window)
        self.main_window = main_window
        self.init_ui()

    def init_ui(self):
        layout = QVBoxLayout(self)
        layout.setSpacing(18)

        self.cam_ip_input = QLineEdit("")
        self.cam_exposure_input = QLineEdit("")
        self.cam_width_input = QLineEdit("")
        self.cam_height_input = QLineEdit("")
        self.cam_offset_x_input = QLineEdit("")
        self.cam_offset_y_input = QLineEdit("")

        self.cam_exposure_input.setPlaceholderText("必填，例如 20000")
        self.cam_width_input.setPlaceholderText("可选")
        self.cam_height_input.setPlaceholderText("可选")
        self.cam_offset_x_input.setPlaceholderText("可选")
        self.cam_offset_y_input.setPlaceholderText("可选")

        form = QFormLayout()
        form.addRow("IP:", self.cam_ip_input)
        form.addRow("ExposureTime:", self.cam_exposure_input)
        form.addRow("Width:", self.cam_width_input)
        form.addRow("Height:", self.cam_height_input)
        form.addRow("OffsetX:", self.cam_offset_x_input)
        form.addRow("OffsetY:", self.cam_offset_y_input)
        layout.addLayout(form)

        self.cam_btn = QPushButton("连接相机")
        self.cam_btn.setCheckable(True)
        self.cam_btn.clicked.connect(self.main_window.connect_or_disconnect_camera)

        self.apply_camera_btn = QPushButton("应用相机参数")
        self.apply_camera_btn.clicked.connect(self.main_window.apply_camera_params_clicked)

        self.preview_btn = QPushButton("采集预览图")
        self.preview_btn.clicked.connect(self.main_window.capture_preview)
        self.export_camera_btn = QPushButton("导出相机参数 JSON")
        self.export_camera_btn.clicked.connect(self.main_window.export_camera_params_json)
        self.confirm_btn = QPushButton("确定，进入压板参数确定")
        self.confirm_btn.clicked.connect(self.main_window.confirm_camera_debug)

        button_layout = QHBoxLayout()
        button_layout.setSpacing(14)
        button_layout.addWidget(self.cam_btn)
        button_layout.addWidget(self.apply_camera_btn)
        button_layout.addWidget(self.preview_btn)
        layout.addLayout(button_layout)
        layout.addWidget(self.export_camera_btn)
        layout.addWidget(self.confirm_btn)

        self.camera_status_label = QLabel("相机参数: 未应用")
        self.camera_status_label.setWordWrap(True)
        layout.addWidget(self.camera_status_label)
        layout.addStretch(1)
