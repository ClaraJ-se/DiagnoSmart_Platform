package com.example.diagnosmart.data.source.local.patient

import androidx.room.Dao
import androidx.room.Insert
import androidx.room.OnConflictStrategy
import androidx.room.Query
import androidx.room.Upsert
import kotlinx.coroutines.flow.Flow

/**
 * for patients
 * */

@Dao
interface PatientDao {
    @Query("SELECT * FROM patient ORDER BY id")
    fun observeAll(): Flow<List<LocalPatient>>

    @Upsert
    fun upsert(patient: LocalPatient)

    @Query("SELECT * FROM patient WHERE roomNumber = :roomNumber ORDER BY id")
    fun getPatientsByRoom(roomNumber: String): Flow<List<LocalPatient>>

    @Query("SELECT * FROM patient WHERE id = :patientId")
    fun getPatient(patientId: Long): LocalPatient

    @Insert(onConflict = OnConflictStrategy.REPLACE)
    suspend fun insert(patient: LocalPatient): Long

    @Insert(onConflict = OnConflictStrategy.REPLACE)
    suspend fun insertAll(patients: List<LocalPatient>)

    @Query("DELETE FROM patient")
    suspend fun deleteAll()
}
