import { ComponentFixture, TestBed } from '@angular/core/testing';

import { Document } from './document';

describe('Document', () => {
  let component: Document;
  let fixture: ComponentFixture<Document>;

  beforeEach(async () => {
    await TestBed.configureTestingModule({
      imports: [Document]
    })
    .compileComponents();

    fixture = TestBed.createComponent(Document);
    component = fixture.componentInstance;
    fixture.detectChanges();
  });

  it('should create', () => {
    expect(component).toBeTruthy();
  });
});
