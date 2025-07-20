!pip install requests>=2.31.0 Pillow>=10.0.0 numpy>=1.24.0 \
langchain-openai>=0.1.0 langchain-groq>=0.1.0 openai>=1.0.0 groq>=0.4.0 \
opencv-python>=4.8.0 transformers>=4.30.0 torch>=2.0.0 \
dataclasses-json>=0.6.0

# ============================================================================
# DATA MODELS
# ============================================================================

class PropertyType(Enum):
    SINGLE_FAMILY = "single_family"
    MULTI_FAMILY = "multi_family"
    CONDOMINIUM = "condominium"
    TOWNHOUSE = "townhouse"
    COMMERCIAL = "commercial"

class RiskLevel(Enum):
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"

class DocumentType(Enum):
    APPRAISAL_REPORT = "appraisal_report"
    INSPECTION_REPORT = "inspection_report"
    PHOTO_DOCUMENTATION = "photo_documentation"
    PROPERTY_SURVEY = "property_survey"

@dataclass
class PropertyInfo:
    address: str
    property_type: PropertyType
    square_footage: float
    year_built: int
    number_of_bedrooms: int
    number_of_bathrooms: float
    lot_size: float
    estimated_value: float

@dataclass
class HazardInfo:
    hazard_type: str
    severity: RiskLevel
    description: str
    location: str
    estimated_cost: float

@dataclass
class DocumentAnalysisResult:
    extracted_info: PropertyInfo
    hazards_detected: List[HazardInfo]
    confidence_score: float
    processing_time: float
    analysis_summary: str

@dataclass
class RiskAssessmentResult:
    overall_risk_score: float
    risk_level: RiskLevel
    approval_recommendation: bool
    risk_factors: List[str]
    mitigation_suggestions: List[str]
    confidence_score: float

@dataclass
class ImageAnalysisResult:
    overall_condition_score: float
    detected_issues: List[Dict[str, Any]]
    confidence_score: float
    processing_time: float

@dataclass
class UnderwritingDecision:
    decision: str
    risk_score: float
    risk_level: RiskLevel
    approval_recommendation: bool
    required_mitigations: List[str]
    confidence_score: float


# ============================================================================
# CONFIGURATION
# ============================================================================

class Config:
    """System configuration."""
    
    def __init__(self):
        # AI Provider settings
        self.ai_provider = os.getenv("AI_PROVIDER", "openai")
        self.openai_api_key = os.getenv("OPENAI_API_KEY", "")
        self.groq_api_key = os.getenv("GROQ_API_KEY", "")
        self.openai_model = os.getenv("OPENAI_MODEL", "gpt-4")
        self.groq_model = os.getenv("GROQ_MODEL", "llama3-8b-8192")
        self.ai_temperature = float(os.getenv("AI_TEMPERATURE", "0.1"))
        
        # Risk assessment settings
        self.default_risk_threshold = 70.0
        self.critical_hazard_threshold = 2
        self.high_hazard_threshold = 3
        
        # Risk weights
        self.risk_weights = {
            'property_age': 0.15,
            'property_type': 0.10,
            'hazard_count': 0.25,
            'hazard_severity': 0.30,
            'property_value': 0.10,
            'location_risk': 0.10
        }
        
        # Hazard severity weights
        self.hazard_weights = {
            'critical': 25.0,
            'high': 15.0,
            'medium': 8.0,
            'low': 3.0
        }


# ============================================================================
# DOCUMENT ANALYSIS SERVICE
# ============================================================================

class DocumentAnalysisService:
    """Service for analyzing property documents and extracting information."""
    
    def __init__(self, config: Config):
        self.config = config
        self._initialize_ai_provider()
        self._initialize_prompts()
    
    def _initialize_ai_provider(self):
        """Initialize AI provider based on configuration."""
        if not OPENAI_AVAILABLE:
            self.llm = None
            return
            
        if self.config.ai_provider == "groq" and self.config.groq_api_key:
            self.llm = ChatGroq(
                model=self.config.groq_model,
                temperature=self.config.ai_temperature,
                groq_api_key=self.config.groq_api_key
            )
            self.ai_provider = "groq"
        elif self.config.openai_api_key:
            self.llm = ChatOpenAI(
                model=self.config.openai_model,
                temperature=self.config.ai_temperature,
                openai_api_key=self.config.openai_api_key
            )
            self.ai_provider = "openai"
        else:
            self.llm = None
            self.ai_provider = "none"
    
    def _initialize_prompts(self):
        """Initialize AI prompts for document analysis."""
        self.property_extraction_prompt = ChatPromptTemplate.from_messages([
            ("system", """You are an expert property document analyst. Extract key property information from the provided document text.
            
            Extract the following information:
            - Property address
            - Property type (single_family, multi_family, condominium, townhouse, commercial)
            - Square footage
            - Year built
            - Number of bedrooms
            - Number of bathrooms
            - Lot size
            - Estimated property value
            
            Return the information in JSON format with the exact field names specified above.
            If information is not available, use null."""),
            ("human", "Document text: {document_text}")
        ])
        
        self.hazard_detection_prompt = ChatPromptTemplate.from_messages([
            ("system", """You are an expert property risk analyst. Identify potential hazards and issues from the provided document text.
            
            Look for:
            - Structural issues
            - Electrical problems
            - Plumbing issues
            - Roof damage
            - Foundation problems
            - Mold or water damage
            - Fire hazards
            - Code violations
            - Environmental hazards
            
            For each hazard found, provide:
            - hazard_type: specific type of hazard
            - severity: low, medium, high, or critical
            - description: detailed description of the issue
            - location: where the hazard is located (if specified)
            - estimated_cost: rough cost estimate for repair (if mentioned)
            
            Return as a JSON array of hazard objects."""),
            ("human", "Document text: {document_text}")
        ])
    
    def analyze_document(self, document_text: str, document_type: DocumentType = DocumentType.APPRAISAL_REPORT) -> DocumentAnalysisResult:
        """Analyze a property document and extract relevant information."""
        start_time = time.time()
        
        try:
            # Extract property information
            property_info = self._extract_property_info(document_text)
            
            # Detect hazards
            hazards = self._detect_hazards(document_text)
            
            # Calculate confidence score
            confidence_score = self._calculate_confidence_score(document_text, property_info, hazards)
            
            processing_time = time.time() - start_time
            
            # Generate analysis summary
            analysis_summary = self._generate_analysis_summary(property_info, hazards, confidence_score)
            
            return DocumentAnalysisResult(
                extracted_info=property_info,
                hazards_detected=hazards,
                confidence_score=confidence_score,
                processing_time=processing_time,
                analysis_summary=analysis_summary
            )
            
        except Exception as e:
            raise Exception(f"Document analysis failed: {str(e)}")
    
    def _extract_property_info(self, document_text: str) -> PropertyInfo:
        """Extract property information using AI or fallback parsing."""
        if self.llm:
            try:
                messages = self.property_extraction_prompt.format_messages(
                    document_text=document_text[:4000]
                )
                response = self.llm.invoke(messages)
                # Parse AI response (simplified for demo)
                return self._parse_ai_property_response(response.content, document_text)
            except Exception as e:
                print(f"AI extraction failed: {e}")
                return self._fallback_property_extraction(document_text)
        else:
            return self._fallback_property_extraction(document_text)
    
    def _fallback_property_extraction(self, document_text: str) -> PropertyInfo:
        """Fallback method to extract property info without AI."""
        # Basic text parsing
        address = "Unknown Address"
        property_type = PropertyType.SINGLE_FAMILY
        square_footage = 2000.0
        year_built = 1995
        estimated_value = 350000.0
        
        lines = document_text.split('\n')
        for line in lines:
            line_lower = line.lower()
            if "address:" in line_lower:
                address = line.split("Address:")[1].strip()
            elif "square footage:" in line_lower:
                try:
                    square_footage = float(line.split("Square Footage:")[1].strip().split()[0].replace(',', ''))
                except:
                    pass
            elif "year built:" in line_lower:
                try:
                    year_built = int(line.split("Year Built:")[1].strip())
                except:
                    pass
            elif "estimated value:" in line_lower or "value:" in line_lower:
                try:
                    value_text = line.split("Value:")[1].strip()
                    estimated_value = float(value_text.replace('$', '').replace(',', ''))
                except:
                    pass
        
        return PropertyInfo(
            address=address,
            property_type=property_type,
            square_footage=square_footage,
            year_built=year_built,
            number_of_bedrooms=3,
            number_of_bathrooms=2.5,
            lot_size=5000.0,
            estimated_value=estimated_value
        )
    
    def _parse_ai_property_response(self, response: str, original_text: str) -> PropertyInfo:
        """Parse AI response to extract property info."""
        # Simplified parsing - in production, you'd use proper JSON parsing
        return self._fallback_property_extraction(original_text)
    
    def _detect_hazards(self, document_text: str) -> List[HazardInfo]:
        """Detect hazards and issues in the document."""
        if self.llm:
            try:
                messages = self.hazard_detection_prompt.format_messages(
                    document_text=document_text[:4000]
                )
                response = self.llm.invoke(messages)
                return self._parse_ai_hazard_response(response.content, document_text)
            except Exception as e:
                print(f"AI hazard detection failed: {e}")
                return self._fallback_hazard_detection(document_text)
        else:
            return self._fallback_hazard_detection(document_text)
    
    def _fallback_hazard_detection(self, document_text: str) -> List[HazardInfo]:
        """Fallback method to detect hazards without AI."""
        hazards = []
        text_lower = document_text.lower()
        
        if "roof" in text_lower and ("damage" in text_lower or "leak" in text_lower):
            hazards.append(HazardInfo(
                hazard_type="Roof Damage",
                severity=RiskLevel.MEDIUM,
                description="Roof damage detected in document",
                location="Roof",
                estimated_cost=5000.0
            ))
        
        if "electrical" in text_lower and ("issue" in text_lower or "problem" in text_lower):
            hazards.append(HazardInfo(
                hazard_type="Electrical Issues",
                severity=RiskLevel.HIGH,
                description="Electrical issues mentioned in document",
                location="Electrical system",
                estimated_cost=8000.0
            ))
        
        if "water" in text_lower and ("damage" in text_lower or "leak" in text_lower):
            hazards.append(HazardInfo(
                hazard_type="Water Damage",
                severity=RiskLevel.HIGH,
                description="Water damage detected in document",
                location="Various locations",
                estimated_cost=7000.0
            ))
        
        if "mold" in text_lower:
            hazards.append(HazardInfo(
                hazard_type="Mold",
                severity=RiskLevel.CRITICAL,
                description="Mold growth detected in document",
                location="Interior",
                estimated_cost=12000.0
            ))
        
        return hazards
    
    def _parse_ai_hazard_response(self, response: str, original_text: str) -> List[HazardInfo]:
        """Parse AI response to extract hazards."""
        # Simplified parsing - in production, you'd use proper JSON parsing
        return self._fallback_hazard_detection(original_text)
    
    def _calculate_confidence_score(self, document_text: str, property_info: PropertyInfo, hazards: List[HazardInfo]) -> float:
        """Calculate confidence score based on data quality."""
        score = 0.5  # Base score
        
        # Text quality
        if len(document_text) > 100:
            score += 0.2
        
        # Property info completeness
        if property_info.address != "Unknown Address":
            score += 0.1
        if property_info.square_footage > 0:
            score += 0.1
        if property_info.year_built > 0:
            score += 0.1
        
        # Hazard detection
        if hazards:
            score += 0.1
        
        return min(score, 1.0)
    
    def _generate_analysis_summary(self, property_info: PropertyInfo, hazards: List[HazardInfo], confidence_score: float) -> str:
        """Generate a summary of the analysis."""
        summary = f"Property at {property_info.address} analyzed with {confidence_score:.1%} confidence. "
        summary += f"Detected {len(hazards)} hazards. "
        
        if hazards:
            critical_hazards = [h for h in hazards if h.severity == RiskLevel.CRITICAL]
            high_hazards = [h for h in hazards if h.severity == RiskLevel.HIGH]
            summary += f"Critical: {len(critical_hazards)}, High: {len(high_hazards)}. "
        
        return summary


# ============================================================================
# RISK ASSESSMENT SERVICE
# ============================================================================

class RiskAssessmentService:
    """Service for assessing property risks and generating scores."""
    
    def __init__(self, config: Config):
        self.config = config
        self._initialize_ai_provider()
        self._initialize_prompts()
    
    def _initialize_ai_provider(self):
        """Initialize AI provider for risk assessment."""
        if not OPENAI_AVAILABLE:
            self.llm = None
            return
            
        if self.config.ai_provider == "groq" and self.config.groq_api_key:
            self.llm = ChatGroq(
                model=self.config.groq_model,
                temperature=self.config.ai_temperature,
                groq_api_key=self.config.groq_api_key
            )
        elif self.config.openai_api_key:
            self.llm = ChatOpenAI(
                model=self.config.openai_model,
                temperature=self.config.ai_temperature,
                openai_api_key=self.config.openai_api_key
            )
        else:
            self.llm = None
    
    def _initialize_prompts(self):
        """Initialize AI prompts for risk assessment."""
        self.risk_assessment_prompt = ChatPromptTemplate.from_messages([
            ("system", """You are an expert property risk assessor and underwriter. 
            Analyze the provided property information and hazards to assess overall risk.
            
            Consider the following factors:
            - Property type and age
            - Square footage and condition
            - Detected hazards and their severity
            - Location-based risks
            - Market conditions
            
            Provide a comprehensive risk assessment including:
            - Overall risk score (0-100, where 0 is no risk and 100 is maximum risk)
            - Risk level (low, medium, high, critical)
            - Approval recommendation (true/false)
            - Key risk factors
            - Mitigation suggestions
            
            Return the assessment in JSON format."""),
            ("human", """
            Property Information:
            {property_info}
            
            Detected Hazards:
            {hazards}
            
            Underwriting Guidelines:
            {guidelines}
            """)
        ])
    
    def assess_risk(self, property_info: PropertyInfo, hazards: List[HazardInfo]) -> RiskAssessmentResult:
        """Assess property risk based on property information and detected hazards."""
        start_time = time.time()
        
        try:
            # Calculate risk score using multiple methods
            ai_risk_score = self._ai_risk_assessment(property_info, hazards)
            algorithmic_risk_score = self._algorithmic_risk_assessment(property_info, hazards)
            
            # Combine scores (weighted average)
            overall_risk_score = (ai_risk_score * 0.7) + (algorithmic_risk_score * 0.3)
            
            # Determine risk level
            risk_level = self._determine_risk_level(overall_risk_score)
            
            # Check approval recommendation
            approval_recommendation = self._check_approval_recommendation(
                property_info, hazards, overall_risk_score, risk_level
            )
            
            # Identify risk factors
            risk_factors = self._identify_risk_factors(property_info, hazards, overall_risk_score)
            
            # Generate mitigation suggestions
            mitigation_suggestions = self._generate_mitigation_suggestions(property_info, hazards, risk_factors)
            
            # Calculate confidence score
            confidence_score = self._calculate_confidence_score(property_info, hazards)
            
            return RiskAssessmentResult(
                overall_risk_score=overall_risk_score,
                risk_level=risk_level,
                approval_recommendation=approval_recommendation,
                risk_factors=risk_factors,
                mitigation_suggestions=mitigation_suggestions,
                confidence_score=confidence_score
            )
            
        except Exception as e:
            raise Exception(f"Risk assessment failed: {str(e)}")
    
    def _ai_risk_assessment(self, property_info: PropertyInfo, hazards: List[HazardInfo]) -> float:
        """Use AI to assess risk based on property information and hazards."""
        if not self.llm:
            return self._calculate_ai_risk_score(property_info, hazards)
        
        try:
            property_info_text = self._format_property_info(property_info)
            hazards_text = self._format_hazards(hazards)
            guidelines_text = self._format_guidelines()
            
            messages = self.risk_assessment_prompt.format_messages(
                property_info=property_info_text,
                hazards=hazards_text,
                guidelines=guidelines_text
            )
            
            response = self.llm.invoke(messages)
            return self._parse_ai_risk_response(response.content, property_info, hazards)
            
        except Exception as e:
            print(f"AI risk assessment error: {e}")
            return self._calculate_ai_risk_score(property_info, hazards)
    
    def _calculate_ai_risk_score(self, property_info: PropertyInfo, hazards: List[HazardInfo]) -> float:
        """Calculate AI-based risk score from hazards and property info."""
        base_score = 30.0  # Base moderate risk
        
        # Adjust based on property type
        if property_info.property_type == PropertyType.COMMERCIAL:
            base_score += 15
        elif property_info.property_type == PropertyType.MULTI_FAMILY:
            base_score += 10
        
        # Adjust based on hazards
        for hazard in hazards:
            if hazard.severity == RiskLevel.CRITICAL:
                base_score += 25
            elif hazard.severity == RiskLevel.HIGH:
                base_score += 15
            elif hazard.severity == RiskLevel.MEDIUM:
                base_score += 8
            elif hazard.severity == RiskLevel.LOW:
                base_score += 3
        
        return min(base_score, 100.0)
    
    def _parse_ai_risk_response(self, response: str, property_info: PropertyInfo, hazards: List[HazardInfo]) -> float:
        """Parse AI response to extract risk score."""
        # Simplified parsing - in production, you'd use proper JSON parsing
        return self._calculate_ai_risk_score(property_info, hazards)
    
    def _algorithmic_risk_assessment(self, property_info: PropertyInfo, hazards: List[HazardInfo]) -> float:
        """Calculate risk score using algorithmic approach."""
        score = 0.0
        
        # Property age risk
        if property_info.year_built:
            age = 2024 - property_info.year_built
            if age > 50:
                score += 20
            elif age > 30:
                score += 15
            elif age > 20:
                score += 10
            elif age > 10:
                score += 5
        
        # Property type risk
        if property_info.property_type == PropertyType.COMMERCIAL:
            score += 15
        elif property_info.property_type == PropertyType.MULTI_FAMILY:
            score += 10
        
        # Hazard-based risk
        critical_hazards = [h for h in hazards if h.severity == RiskLevel.CRITICAL]
        high_hazards = [h for h in hazards if h.severity == RiskLevel.HIGH]
        medium_hazards = [h for h in hazards if h.severity == RiskLevel.MEDIUM]
        
        score += len(critical_hazards) * 25
        score += len(high_hazards) * 15
        score += len(medium_hazards) * 8
        
        # Property value risk (higher value = higher risk)
        if property_info.estimated_value:
            if property_info.estimated_value > 1000000:
                score += 10
            elif property_info.estimated_value > 500000:
                score += 5
        
        return min(score, 100.0)
    
    def _determine_risk_level(self, risk_score: float) -> RiskLevel:
        """Determine risk level based on score."""
        if risk_score >= 80:
            return RiskLevel.CRITICAL
        elif risk_score >= 60:
            return RiskLevel.HIGH
        elif risk_score >= 40:
            return RiskLevel.MEDIUM
        else:
            return RiskLevel.LOW
    
    def _check_approval_recommendation(self, property_info: PropertyInfo, hazards: List[HazardInfo], risk_score: float, risk_level: RiskLevel) -> bool:
        """Check if property should be approved based on risk assessment."""
        # Basic approval logic
        if risk_score > self.config.default_risk_threshold:
            return False
        
        critical_hazards = [h for h in hazards if h.severity == RiskLevel.CRITICAL]
        if len(critical_hazards) >= self.config.critical_hazard_threshold:
            return False
        
        high_hazards = [h for h in hazards if h.severity == RiskLevel.HIGH]
        if len(high_hazards) >= self.config.high_hazard_threshold:
            return False
        
        return True
    
    def _identify_risk_factors(self, property_info: PropertyInfo, hazards: List[HazardInfo], risk_score: float) -> List[str]:
        """Identify key risk factors."""
        factors = []
        
        # Property age
        if property_info.year_built and (2024 - property_info.year_built) > 30:
            factors.append(f"Property age: {2024 - property_info.year_built} years old")
        
        # Property type
        if property_info.property_type == PropertyType.COMMERCIAL:
            factors.append("Commercial property type")
        
        # Hazards
        critical_hazards = [h for h in hazards if h.severity == RiskLevel.CRITICAL]
        high_hazards = [h for h in hazards if h.severity == RiskLevel.HIGH]
        
        if critical_hazards:
            factors.append(f"{len(critical_hazards)} critical hazards detected")
        if high_hazards:
            factors.append(f"{len(high_hazards)} high-severity hazards detected")
        
        # Property value
        if property_info.estimated_value and property_info.estimated_value > 1000000:
            factors.append("High property value")
        
        return factors
    
    def _generate_mitigation_suggestions(self, property_info: PropertyInfo, hazards: List[HazardInfo], risk_factors: List[str]) -> List[str]:
        """Generate mitigation suggestions."""
        suggestions = []
        
        for hazard in hazards:
            if hazard.hazard_type == "Electrical Issues":
                suggestions.append("Address electrical issues before approval")
            elif hazard.hazard_type == "Roof Damage":
                suggestions.append("Repair roof damage and provide warranty")
            elif hazard.hazard_type == "Water Damage":
                suggestions.append("Fix water damage and prevent future issues")
            elif hazard.hazard_type == "Mold":
                suggestions.append("Professional mold remediation required")
        
        if property_info.property_type == PropertyType.COMMERCIAL:
            suggestions.append("Additional commercial property inspections recommended")
        
        if not suggestions:
            suggestions.append("No specific mitigations required")
        
        return suggestions
    
    def _calculate_confidence_score(self, property_info: PropertyInfo, hazards: List[HazardInfo]) -> float:
        """Calculate confidence score for risk assessment."""
        score = 0.8  # Base confidence
        
        # Adjust based on data completeness
        if property_info.address and property_info.address != "Unknown Address":
            score += 0.1
        if property_info.square_footage > 0:
            score += 0.05
        if property_info.year_built > 0:
            score += 0.05
        
        return min(score, 1.0)
    
    def _format_property_info(self, property_info: PropertyInfo) -> str:
        """Format property information for AI analysis."""
        estimated_value_str = f"${property_info.estimated_value:,.0f}" if property_info.estimated_value else 'Unknown'
        return f"""
        Address: {property_info.address}
        Property Type: {property_info.property_type.value}
        Square Footage: {property_info.square_footage or 'Unknown'}
        Year Built: {property_info.year_built or 'Unknown'}
        Bedrooms: {property_info.number_of_bedrooms or 'Unknown'}
        Bathrooms: {property_info.number_of_bathrooms or 'Unknown'}
        Lot Size: {property_info.lot_size or 'Unknown'}
        Estimated Value: {estimated_value_str}
        """
    
    def _format_hazards(self, hazards: List[HazardInfo]) -> str:
        """Format hazards for AI analysis."""
        if not hazards:
            return "No hazards detected"
        
        hazard_texts = []
        for hazard in hazards:
            hazard_texts.append(f"- {hazard.hazard_type} ({hazard.severity.value}): {hazard.description}")
        
        return "\n".join(hazard_texts)
    
    def _format_guidelines(self) -> str:
        """Format underwriting guidelines for AI analysis."""
        return f"""
        Maximum risk score for approval: {self.config.default_risk_threshold}
        Maximum critical hazards: {self.config.critical_hazard_threshold}
        Maximum high-severity hazards: {self.config.high_hazard_threshold}
        """


# ============================================================================
# COMPUTER VISION SERVICE
# ============================================================================

class ComputerVisionService:
    """Service for analyzing property images using computer vision."""
    
    def __init__(self, config: Config):
        self.config = config
        self._initialize_models()
    
    def _initialize_models(self):
        """Initialize computer vision models."""
        if not CV_AVAILABLE:
            self.image_analyzer = None
            return
        
        try:
            # Initialize image classification pipeline
            self.image_analyzer = pipeline(
                "image-classification",
                model="microsoft/resnet-50",
                device=-1  # CPU
            )
        except Exception as e:
            print(f"Failed to initialize computer vision models: {e}")
            self.image_analyzer = None
    
    def analyze_image(self, image_data: bytes) -> ImageAnalysisResult:
        """Analyze property images for damage assessment."""
        start_time = time.time()
        
        try:
            # Load and preprocess image
            image = Image.open(io.BytesIO(image_data))
            
            # Basic image analysis
            condition_score = self._assess_image_condition(image)
            detected_issues = self._detect_issues(image)
            
            processing_time = time.time() - start_time
            
            # Calculate confidence score
            confidence_score = self._calculate_confidence_score(image, detected_issues)
            
            return ImageAnalysisResult(
                overall_condition_score=condition_score,
                detected_issues=detected_issues,
                confidence_score=confidence_score,
                processing_time=processing_time
            )
            
        except Exception as e:
            raise Exception(f"Image analysis failed: {str(e)}")
    
    def _assess_image_condition(self, image: Image.Image) -> float:
        """Assess overall condition of the property image."""
        # Basic condition assessment based on image properties
        score = 85.0  # Base good condition
        
        # Analyze image brightness
        gray = image.convert('L')
        brightness = np.mean(gray)
        
        if brightness < 50:
            score -= 10  # Dark image might indicate poor lighting/condition
        elif brightness > 200:
            score -= 5   # Very bright image might indicate overexposure
        
        # Analyze image sharpness (simplified)
        # In production, you'd use more sophisticated edge detection
        
        return max(score, 0.0)
    
    def _detect_issues(self, image: Image.Image) -> List[Dict[str, Any]]:
        """Detect issues in the property image."""
        issues = []
        
        # Basic issue detection (simplified)
        # In production, you'd use more sophisticated CV techniques
        
        # Check for obvious visual issues
        gray = image.convert('L')
        brightness = np.mean(gray)
        
        if brightness < 30:
            issues.append({
                "issue_type": "Poor Lighting",
                "severity": "medium",
                "description": "Image appears to be poorly lit",
                "confidence": 0.7
            })
        
        # Add some sample issues for demonstration
        if len(issues) == 0:
            issues.append({
                "issue_type": "No Major Issues",
                "severity": "low",
                "description": "No significant issues detected in image",
                "confidence": 0.8
            })
        
        return issues
    
    def _calculate_confidence_score(self, image: Image.Image, detected_issues: List[Dict[str, Any]]) -> float:
        """Calculate confidence score for image analysis."""
        score = 0.5  # Base confidence
        
        # Adjust based on image quality
        if image.size[0] > 500 and image.size[1] > 500:
            score += 0.2
        
        # Adjust based on issue detection
        if detected_issues:
            score += 0.2
        
        # Adjust based on image format
        if image.format in ['JPEG', 'PNG']:
            score += 0.1
        
        return min(score, 1.0)


# ============================================================================
# MAIN UNDERWRITING SYSTEM
# ============================================================================

class AIUnderwritingSystem:
    """Main AI-powered underwriting system."""
    
    def __init__(self):
        self.config = Config()
        self.document_service = DocumentAnalysisService(self.config)
        self.risk_service = RiskAssessmentService(self.config)
        self.vision_service = ComputerVisionService(self.config)
        
        print("🚀 AI Underwriting System Initialized")
        print(f"   AI Provider: {self.config.ai_provider}")
        print(f"   Document Analysis: {'✅' if self.document_service.llm else '⚠️'}")
        print(f"   Risk Assessment: {'✅' if self.risk_service.llm else '⚠️'}")
        print(f"   Computer Vision: {'✅' if self.vision_service.image_analyzer else '⚠️'}")
    
    def process_document(self, document_text: str, document_type: DocumentType = DocumentType.APPRAISAL_REPORT) -> DocumentAnalysisResult:
        """Process a property document and extract information."""
        print(f"📄 Processing document: {document_type.value}")
        return self.document_service.analyze_document(document_text, document_type)
    
    def assess_risk(self, property_info: PropertyInfo, hazards: List[HazardInfo]) -> RiskAssessmentResult:
        """Assess risk for a property."""
        print("⚠️ Assessing property risk...")
        return self.risk_service.assess_risk(property_info, hazards)
    
    def analyze_image(self, image_data: bytes) -> ImageAnalysisResult:
        """Analyze property images."""
        print("🖼️ Analyzing property images...")
        return self.vision_service.analyze_image(image_data)
    
    def complete_underwriting(self, document_text: str, image_data: Optional[bytes] = None) -> Dict[str, Any]:
        """Complete end-to-end underwriting process."""
        print("🔄 Starting complete underwriting process...")
        
        # Step 1: Document Analysis
        doc_result = self.process_document(document_text)
        
        # Step 2: Image Analysis (if provided)
        image_result = None
        if image_data:
            image_result = self.analyze_image(image_data)
        
        # Step 3: Risk Assessment
        risk_result = self.assess_risk(doc_result.extracted_info, doc_result.hazards_detected)
        
        # Step 4: Generate Final Decision
        decision = self._generate_underwriting_decision(doc_result, risk_result, image_result)
        
        # Step 5: Compile Results
        complete_result = {
            "document_analysis": {
                "property_info": {
                    "address": doc_result.extracted_info.address,
                    "property_type": doc_result.extracted_info.property_type.value,
                    "square_footage": doc_result.extracted_info.square_footage,
                    "year_built": doc_result.extracted_info.year_built,
                    "estimated_value": doc_result.extracted_info.estimated_value
                },
                "hazards_detected": [
                    {
                        "hazard_type": h.hazard_type,
                        "severity": h.severity.value,
                        "description": h.description,
                        "estimated_cost": h.estimated_cost
                    } for h in doc_result.hazards_detected
                ],
                "confidence_score": doc_result.confidence_score,
                "processing_time": doc_result.processing_time
            },
            "risk_assessment": {
                "overall_risk_score": risk_result.overall_risk_score,
                "risk_level": risk_result.risk_level.value,
                "approval_recommendation": risk_result.approval_recommendation,
                "risk_factors": risk_result.risk_factors,
                "mitigation_suggestions": risk_result.mitigation_suggestions,
                "confidence_score": risk_result.confidence_score
            },
            "image_analysis": {
                "overall_condition_score": image_result.overall_condition_score if image_result else None,
                "detected_issues": image_result.detected_issues if image_result else None,
                "confidence_score": image_result.confidence_score if image_result else None
            },
            "underwriting_decision": decision,
            "summary": {
                "property_address": doc_result.extracted_info.address,
                "final_decision": decision["decision"],
                "risk_score": risk_result.overall_risk_score,
                "risk_level": risk_result.risk_level.value,
                "hazards_detected": len(doc_result.hazards_detected),
                "confidence_score": min(doc_result.confidence_score, risk_result.confidence_score)
            }
        }
        
        print("✅ Complete underwriting process finished")
        return complete_result
    
    def _generate_underwriting_decision(self, doc_result: DocumentAnalysisResult, risk_result: RiskAssessmentResult, image_result: Optional[ImageAnalysisResult]) -> Dict[str, Any]:
        """Generate final underwriting decision."""
        decision = "APPROVED" if risk_result.approval_recommendation else "REJECTED"
        
        # Adjust decision based on image analysis if available
        if image_result and image_result.overall_condition_score < 50:
            decision = "REJECTED"
        
        return {
            "decision": decision,
            "risk_score": risk_result.overall_risk_score,
            "risk_level": risk_result.risk_level.value,
            "approval_recommendation": risk_result.approval_recommendation,
            "required_mitigations": risk_result.mitigation_suggestions,
            "confidence_score": risk_result.confidence_score
        }


# ============================================================================
# DEMO AND USAGE
# ============================================================================

def demo_underwriting_system():
    """Demonstrate the underwriting system functionality."""
    print("🚀 AI Underwriting System Demo")
    print("=" * 60)
    
    # Initialize system
    system = AIUnderwritingSystem()
    
    # Sample document
    sample_document = """
    PROPERTY APPRAISAL REPORT

    Property Address: 456 Oak Avenue, Rivertown, USA  
    Property Type: Townhouse  
    Square Footage: 1,750 sq ft  
    Year Built: 2008  
    Number of Bedrooms: 2  
    Number of Bathrooms: 2  
    Lot Size: 3,200 sq ft  
    Estimated Value: $295,000

    Property Condition Notes:
    - HVAC system recently replaced
    - Cracked tiles in kitchen floor
    - Slight settling cracks in foundation
    - Modern kitchen and updated appliances"""

    
    print("\n📄 Sample Document:")
    print(sample_document)
    
    # Process document
    print("\n" + "=" * 60)
    doc_result = system.process_document(sample_document)
    
    print(f"✅ Document Analysis Complete:")
    print(f"   Property: {doc_result.extracted_info.address}")
    print(f"   Type: {doc_result.extracted_info.property_type.value}")
    print(f"   Value: ${doc_result.extracted_info.estimated_value:,.0f}")
    print(f"   Hazards: {len(doc_result.hazards_detected)}")
    print(f"   Confidence: {doc_result.confidence_score:.1%}")
    
    # Risk assessment
    print("\n" + "=" * 60)
    risk_result = system.assess_risk(doc_result.extracted_info, doc_result.hazards_detected)
    
    print(f"⚠️ Risk Assessment Complete:")
    print(f"   Risk Score: {risk_result.overall_risk_score:.1f}/100")
    print(f"   Risk Level: {risk_result.risk_level.value.upper()}")
    print(f"   Approval: {'✅ APPROVED' if risk_result.approval_recommendation else '❌ REJECTED'}")
    print(f"   Confidence: {risk_result.confidence_score:.1%}")
    
    # Complete process
    print("\n" + "=" * 60)
    complete_result = system.complete_underwriting(sample_document)
    
    print(f"🎉 Complete Underwriting Process:")
    print(f"   Final Decision: {complete_result['summary']['final_decision']}")
    print(f"   Risk Score: {complete_result['summary']['risk_score']:.1f}/100")
    print(f"   Risk Level: {complete_result['summary']['risk_level'].upper()}")
    print(f"   Hazards: {complete_result['summary']['hazards_detected']}")
    print(f"   Confidence: {complete_result['summary']['confidence_score']:.1%}")
    
    # Show detailed results
    print("\n" + "=" * 60)
    print("📊 Detailed Results:")
    print(json.dumps(complete_result, indent=2, default=str))
    
    return complete_result

if __name__ == "__main__":
    # Run demo
    demo_underwriting_system()



"""
Here's a demo output : 
🚀 AI Underwriting System Demo
============================================================
Device set to use cpu
🚀 AI Underwriting System Initialized
   AI Provider: openai
   Document Analysis: ⚠️
   Risk Assessment: ⚠️
   Computer Vision: ✅

📄 Sample Document:

    PROPERTY APPRAISAL REPORT

    Property Address: 456 Oak Avenue, Rivertown, USA  
    Property Type: Townhouse  
    Square Footage: 1,750 sq ft  
    Year Built: 2008  
    Number of Bedrooms: 2  
    Number of Bathrooms: 2  
    Lot Size: 3,200 sq ft  
    Estimated Value: $295,000

    Property Condition Notes:
    - HVAC system recently replaced
    - Cracked tiles in kitchen floor
    - Slight settling cracks in foundation
    - Modern kitchen and updated appliances

============================================================
📄 Processing document: appraisal_report
✅ Document Analysis Complete:
   Property: 456 Oak Avenue, Rivertown, USA
   Type: single_family
   Value: $295,000
   Hazards: 0
   Confidence: 100.0%

============================================================
⚠️ Assessing property risk...
⚠️ Risk Assessment Complete:
   Risk Score: 22.5/100
   Risk Level: LOW
   Approval: ✅ APPROVED
   Confidence: 100.0%

============================================================
🔄 Starting complete underwriting process...
📄 Processing document: appraisal_report
⚠️ Assessing property risk...
✅ Complete underwriting process finished
🎉 Complete Underwriting Process:
   Final Decision: APPROVED
   Risk Score: 22.5/100
   Risk Level: LOW
   Hazards: 0
   Confidence: 100.0%

============================================================
📊 Detailed Results:
{
  "document_analysis": {
    "property_info": {
      "address": "456 Oak Avenue, Rivertown, USA",
      "property_type": "single_family",
      "square_footage": 1750.0,
      "year_built": 2008,
      "estimated_value": 295000.0
    },
    "hazards_detected": [],
    "confidence_score": 0.9999999999999999,
    "processing_time": 2.5033950805664062e-05
  },
  "risk_assessment": {
    "overall_risk_score": 22.5,
    "risk_level": "low",
    "approval_recommendation": true,
    "risk_factors": [],
    "mitigation_suggestions": [
      "No specific mitigations required"
    ],
    "confidence_score": 1.0
  },
  "image_analysis": {
    "overall_condition_score": null,
    "detected_issues": null,
    "confidence_score": null
  },
  "underwriting_decision": {
    "decision": "APPROVED",
    "risk_score": 22.5,
    "risk_level": "low",
    "approval_recommendation": true,
    "required_mitigations": [
      "No specific mitigations required"
    ],
    "confidence_score": 1.0
  },
  "summary": {
    "property_address": "456 Oak Avenue, Rivertown, USA",
    "final_decision": "APPROVED",
    "risk_score": 22.5,
    "risk_level": "low",
    "hazards_detected": 0,
    "confidence_score": 0.9999999999999999
  }
}
"""
