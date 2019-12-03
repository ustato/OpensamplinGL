# -*- coding: utf-8 -*-
# ---
# jupyter:
#   jupytext:
#     text_representation:
#       extension: .py
#       format_name: light
#       format_version: '1.4'
#       jupytext_version: 1.2.4
#   kernelspec:
#     display_name: Python 3
#     language: python
#     name: python3
# ---

# +
# OpenGLとGLFWをインポートします
from OpenGL.GL import *
import glfw


def main():
    # GLFW初期化
    if not glfw.init():
        return

    # ウィンドウを作成
    window = glfw.create_window(640, 480, 'Hello World', None, None)
    if not window:
        glfw.terminate()
        print('Failed to create window')
        return

    # コンテキストを作成します
    glfw.make_context_current(window)

    # OpenGLのバージョン等を表示します
    print('Vendor :', glGetString(GL_VENDOR))
    print('GPU :', glGetString(GL_RENDERER))
    print('OpenGL version :', glGetString(GL_VERSION))

    # ウィンドウを破棄してGLFWを終了します
    glfw.destroy_window(window)
    glfw.terminate()


# Pythonのメイン関数はこんな感じで書きます
if __name__ == "__main__":
    main()
