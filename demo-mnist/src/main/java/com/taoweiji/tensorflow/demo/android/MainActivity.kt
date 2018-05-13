package com.taoweiji.tensorflow.demo.android

import android.graphics.Bitmap
import android.graphics.BitmapFactory
import android.graphics.Color
import android.graphics.Matrix
import android.os.Bundle
import android.support.v7.app.AppCompatActivity
import kotlinx.android.synthetic.main.activity_main.*
import org.tensorflow.contrib.android.TensorFlowInferenceInterface


class MainActivity : AppCompatActivity() {

    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        setContentView(R.layout.activity_main)

        val bitmap = BitmapFactory.decodeResource(resources, R.drawable.test_image)
        val tfi = TensorFlowInferenceInterface(assets, "mnist.pb")
        val inputData = bitmapToFloatArray(bitmap, 28f, 28f)
        tfi.feed("input/x_input", inputData, 1, 784)
        val outputNames = arrayOf("output")
        tfi.run(outputNames)
        // 用于存储模型的输出数据
        val outputs = IntArray(1)
        tfi.fetch(outputNames[0], outputs)

        imageView.setImageBitmap(bitmap)
        textView.text = "结果为：" + outputs[0]
    }

    /**
     * 将bitmap转为（按行优先）一个float数组，并且每个像素点都归一化到0~1之间。
     * @param bitmap 输入被测试的bitmap图片
     * @param rx 将图片缩放到指定的大小（列）->28
     * @param ry 将图片缩放到指定的大小（行）->28
     * @return   返回归一化后的一维float数组 ->28*28
     */
    private fun bitmapToFloatArray(bitmap: Bitmap, rx: Float, ry: Float): FloatArray {
        var height = bitmap.height
        var width = bitmap.width
        // 计算缩放比例
        val scaleWidth = rx / width
        val scaleHeight = ry / height
        val matrix = Matrix()
        matrix.postScale(scaleWidth, scaleHeight)
        val bitmap = Bitmap.createBitmap(bitmap, 0, 0, width, height, matrix, true)
        height = bitmap.height
        width = bitmap.width
        val result = FloatArray(height * width)
        var k = 0
        for (row in 0 until height) {
            for (col in 0 until width) {
                val argb = bitmap.getPixel(col, row)
                val r = Color.red(argb)
                val g = Color.green(argb)
                val b = Color.blue(argb)
                //由于是灰度图，所以r,g,b分量是相等的。
                assert(r == g && g == b)
                result[k++] = r / 255.0f
            }
        }
        return result
    }
}
