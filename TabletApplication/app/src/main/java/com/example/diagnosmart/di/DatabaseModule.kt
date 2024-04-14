package com.example.diagnosmart.di

import android.content.Context
import com.example.diagnosmart.data.source.local.AppDatabase
import com.example.diagnosmart.data.source.local.auth.AuthManager
import com.example.diagnosmart.data.source.local.patient.PatientDao
import com.example.diagnosmart.data.source.local.user.UserDao
import dagger.Module
import dagger.Provides
import dagger.hilt.InstallIn
import dagger.hilt.android.qualifiers.ApplicationContext
import dagger.hilt.components.SingletonComponent
import javax.inject.Singleton

@Module
@InstallIn(SingletonComponent::class)
class DatabaseModule {

    @Singleton
    @Provides
    fun provideAppDatabase(@ApplicationContext context: Context): AppDatabase {
        return AppDatabase.getInstance(context)
    }

    @Provides
    fun providePatientDao(appDatabase: AppDatabase): PatientDao = appDatabase.patientDao()

    @Provides
    fun provideUserDao(appDatabase: AppDatabase): UserDao = appDatabase.userDao()

    @Provides
    fun provideAuthManager(@ApplicationContext context: Context) = AuthManager(context)

}

