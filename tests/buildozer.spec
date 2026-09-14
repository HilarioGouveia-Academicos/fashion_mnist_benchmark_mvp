[app]
title = Fashion Classifier
package.name = fashionclassifier
package.domain = org.test

source.dir = .
source.include_exts = py,png,jpg,kv,atlas

version = 0.1
requirements = python3,kivy,opencv,numpy,requests,urllib3,certifi,charset_normalizer,idna

android.permissions = INTERNET,CAMERA,READ_EXTERNAL_STORAGE,WRITE_EXTERNAL_STORAGE
android.archs = arm64-v8a, armeabi-v7a

# Permite tráfego HTTP sem certificado SSL na sua rede local
android.manifest.application_arguments = android:usesCleartextTraffic="true"