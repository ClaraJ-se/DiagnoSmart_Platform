package com.example.diagnosmart

import android.app.Application
import dagger.hilt.android.HiltAndroidApp

@HiltAndroidApp
class DiagnoSmartApplication : Application() {

    override fun onCreate() {
        super.onCreate()
//        if (BuildConfig.DEBUG) Timber.plant(Timber.DebugTree())
    }

}

// create login page
// create main page
// create data base with mock patients
// create info page
// create result page