package com.brandanalytics.yolov5ncnn;

import android.content.res.AssetManager;
import android.graphics.Bitmap;

import java.nio.ByteBuffer;

public class YoloV5Ncnn
{
    public native boolean Init(AssetManager mgr, ByteBuffer modelBuffer);

    public class Obj
    {
        public float x;
        public float y;
        public float w;
        public float h;
        public String label;
        public float prob;
    }
    public native Obj[] Detect(Bitmap image);

    static {
        System.loadLibrary("nnbackend");
    }
}
