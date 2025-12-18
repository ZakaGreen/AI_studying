from PyQt5 import QtWidgets, uic
import sys

app = QtWidgets.QApplication(sys.argv)
window = uic.loadUi("MyForm.ui")
if window is None:
    print("Ошибка: не удалось загрузить файл MyForm.ui")
    sys.exit(1)
window.show()
sys.exit(app.exec_())

