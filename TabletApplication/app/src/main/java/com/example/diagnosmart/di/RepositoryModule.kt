package com.example.diagnosmart.di


import com.example.diagnosmart.data.source.repo.AccountRepository
import com.example.diagnosmart.data.source.repo.PatientRepository
import com.example.diagnosmart.data.source.repo.impl.AccountRepositoryImpl
import com.example.diagnosmart.data.source.repo.impl.PatientRepositoryImpl
import dagger.Binds
import dagger.Module
import dagger.hilt.InstallIn
import dagger.hilt.components.SingletonComponent
import javax.inject.Singleton

@Module
@InstallIn(SingletonComponent::class)
abstract class RepositoryModule {

    @Singleton
    @Binds
    abstract fun bindAccountRepository(repository: AccountRepositoryImpl): AccountRepository

    @Singleton
    @Binds
    abstract fun bindPatientRepository(repository: PatientRepositoryImpl): PatientRepository

}