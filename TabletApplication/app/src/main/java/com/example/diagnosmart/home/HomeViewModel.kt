package com.example.diagnosmart.home

import androidx.lifecycle.ViewModel
import androidx.lifecycle.viewModelScope
import com.example.diagnosmart.data.source.models.Patient
import com.example.diagnosmart.data.source.repo.PatientRepository
import dagger.hilt.android.lifecycle.HiltViewModel
import kotlinx.coroutines.flow.MutableStateFlow
import kotlinx.coroutines.flow.StateFlow
import kotlinx.coroutines.flow.asStateFlow
import kotlinx.coroutines.flow.update
import kotlinx.coroutines.launch
import javax.inject.Inject

data class HomeTaskUiState(
        val isTaskCompleted: Boolean = false,
        val isLoading: Boolean = false,
        val userMessage: Int? = null,
        val error: String? = null,

        val patients: List<Patient> = emptyList(),

        )

@HiltViewModel
class HomeViewModel @Inject constructor(
        private val patientRepository: PatientRepository
) : ViewModel() {

    private val _uiState = MutableStateFlow(HomeTaskUiState())
    val uiState: StateFlow<HomeTaskUiState> = _uiState.asStateFlow()

    init {
        observePatients()
    }

    private fun observePatients() {
        viewModelScope.launch {
            patientRepository.getMockPatientsStream().collect { patients ->
                _uiState.update {
                    it.copy(
                            patients = patients.reversed()
                    )
                }
            }
        }
    }

}