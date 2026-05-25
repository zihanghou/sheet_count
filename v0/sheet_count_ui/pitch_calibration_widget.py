from PySide6.QtWidgets import QFormLayout, QHBoxLayout, QLabel, QLineEdit, QPushButton, QVBoxLayout, QWidget


class PitchCalibrationWidget(QWidget):
    def __init__(self, main_window):
        super().__init__(main_window)
        self.main_window = main_window
        self.init_ui()

    def init_ui(self):
        layout = QVBoxLayout(self)
        layout.setSpacing(24)

        title = QLabel("压板参数确定")
        title.setStyleSheet("font-size: 20pt; font-weight: 700;")
        layout.addWidget(title)

        self.correct_count_input = QLineEdit("")
        self.correct_count_input.setPlaceholderText("输入当前图片真实数量")

        form = QFormLayout()
        form.addRow("正确数量:", self.correct_count_input)
        layout.addLayout(form)

        self.pitch_capture_btn = QPushButton("采图并加入压板参数样本")
        self.pitch_capture_btn.clicked.connect(self.capture_pitch_sample)

        self.import_camera_btn = QPushButton("导入相机参数 JSON")
        self.import_camera_btn.clicked.connect(self.main_window.import_camera_params_json)

        self.export_btn = QPushButton("导出参数 JSON")
        self.export_btn.clicked.connect(self.main_window.export_params_json)
        self.confirm_btn = QPushButton("确定，进入检测流程")
        self.confirm_btn.clicked.connect(self.main_window.confirm_pitch_calibration)

        button_layout = QHBoxLayout()
        button_layout.setSpacing(14)
        button_layout.addWidget(self.pitch_capture_btn)
        button_layout.addWidget(self.import_camera_btn)
        button_layout.addWidget(self.export_btn)
        layout.addLayout(button_layout)
        layout.addWidget(self.confirm_btn)

        self.pitch_label = QLabel("样本 0 张  |  还需 2 张")
        self.pitch_label.setWordWrap(True)
        self.pitch_label.setStyleSheet(
            """
            QLabel {
                background-color: #F3F4F6;
                border: 1px solid #D1D5DB;
                border-radius: 8px;
                color: #111827;
                font-size: 22pt;
                font-weight: 700;
                padding: 22px;
            }
            """
        )
        layout.addWidget(self.pitch_label)

        self.samples_label = QLabel("每张参数结果: 暂无")
        self.samples_label.setWordWrap(True)
        self.samples_label.setStyleSheet(
            """
            QLabel {
                background-color: #FFFFFF;
                border: 1px solid #D1D5DB;
                border-radius: 8px;
                color: #111827;
                font-size: 16pt;
                font-weight: 600;
                line-height: 150%;
                padding: 18px;
            }
            """
        )
        layout.addWidget(self.samples_label)
        layout.addStretch(1)

    def capture_pitch_sample(self):
        self.main_window.capture_pitch_sample(self.correct_count_input.text())
